import simpy
import networkx as nx
import matplotlib.pyplot as plt
# import random

RANDOM_SEED = 42
LOCAL_MEMORY_SIZE = 4096
GLOBAL_MEMORY_SIZE = 16384


class TileAddress:
    def __init__(self, base, offset, annotation):
        self.base = base # (x, y)
        self.offset = offset # (x, y)
        self.annotation = annotation #e.g. X, Wq, Wk...

    def __str__(self):
        return f"TileAddress(base={self.base}, offset={self.offset}, annotation={self.annotation})"
    
    def __repr__(self):
        return f"TileAddress(base={self.base}, offset={self.offset}, annotation={self.annotation})"
    
    def __hash__(self):
        return hash((self.base, self.offset, self.annotation))
    
    def __eq__(self, other):
        if not isinstance(other, TileAddress):
            return False
        return (self.base == other.base and 
                self.offset == other.offset and 
                self.annotation == other.annotation)

def is_overlap(addrA, addrB):
    if addrA.annotation != addrB.annotation:
        return False
    if (addrA.base[0] < addrB.base[0] + addrB.offset[0] and
            addrA.base[0] + addrA.offset[0] > addrB.base[0] and
            addrA.base[1] < addrB.base[1] + addrB.offset[1] and
            addrA.base[1] + addrA.offset[1] > addrB.base[1]):
        return True
    elif (addrB.base[0] < addrA.base[0] + addrA.offset[0] and
            addrB.base[0] + addrB.offset[0] > addrA.base[0] and
            addrB.base[1] < addrA.base[1] + addrA.offset[1] and
            addrB.base[1] + addrB.offset[1] > addrA.base[1]):
        return True
    return False

def overlap(addrA, addrB):
    # return tile addresses overlap if their base coordinates and offsets overlap
    # otherwise return None 
    if addrA.annotation != addrB.annotation:
        return None
    if (addrA.base[0] < addrB.base[0] + addrB.offset[0] and
            addrA.base[0] + addrA.offset[0] > addrB.base[0] and
            addrA.base[1] < addrB.base[1] + addrB.offset[1] and
            addrA.base[1] + addrA.offset[1] > addrB.base[1]):
        return TileAddress(
            (max(addrA.base[0], addrB.base[0]), max(addrA.base[1], addrB.base[1])),
            (min(addrA.base[0] + addrA.offset[0], addrB.base[0] + addrB.offset[0]) - max(addrA.base[0], addrB.base[0]),
             min(addrA.base[1] + addrA.offset[1], addrB.base[1] + addrB.offset[1]) - max(addrA.base[1], addrB.base[1])),
            addrA.annotation
        )
    elif (addrB.base[0] < addrA.base[0] + addrA.offset[0] and
            addrB.base[0] + addrB.offset[0] > addrA.base[0] and
            addrB.base[1] < addrA.base[1] + addrA.offset[1] and
            addrB.base[1] + addrB.offset[1] > addrA.base[1]):
        return TileAddress(
            (max(addrA.base[0], addrB.base[0]), max(addrA.base[1], addrB.base[1])),
            (min(addrA.base[0] + addrA.offset[0], addrB.base[0] + addrB.offset[0]) - max(addrA.base[0], addrB.base[0]),
             min(addrA.base[1] + addrA.offset[1], addrB.base[1] + addrB.offset[1]) - max(addrA.base[1], addrB.base[1])),
            addrB.annotation
        )
    return None

# class MemoryRequest:
#     def __init__(self, core_idx, type, addr):
#         self.core_idx = core_idx
#         self.type = type # e.g. 'load', 'store'
#         self.addr = addr # TileAddress

# class NoCRequest:
#     def __init__(self, src_core_idx, dest_core_idx, type, addr):
#         self.src_core_idx = src_core_idx
#         self.dest_core_idx = dest_core_idx
#         self.type = type # e.g. 'unicast'
#         self.addr = addr # TileAddress

class SliceMatmul:
    def __init__(self, m, k, n, addrA, addrB, annotationRes):
        self.m = m
        self.k = k
        self.n = n
        self.addrA = addrA
        self.addrB = addrB
        self.addrRes = TileAddress(
            (addrB.base[0], addrA.base[1]),
            (n, m),
            annotationRes
        )
        self.complete_event = None
        self.tile_id = None

    def __str__(self):
        return f"SliceMatmul(m={self.m}, k={self.k}, n={self.n}, addrA={self.addrA}, addrB={self.addrB}, addrRes={self.addrRes})"

    def __repr__(self):
        return f"SliceMatmul(m={self.m}, k={self.k}, n={self.n}, addrA={self.addrA}, addrB={self.addrB}, addrRes={self.addrRes})"

class Matmul:
    def __init__(self, m, k, n, annotations, tile_size):
        self.m = m
        self.k = k
        self.n = n
        self.annotations = annotations # (e.g. {'A': 'X', 'B': 'Wq', 'Res': 'Q'})
        self.tile_size = tile_size
        self.m_pad = (m + tile_size[0] - 1) // tile_size[0] * tile_size[0]
        self.n_pad = (n + tile_size[1] - 1) // tile_size[1] * tile_size[1]
        self.m_split = self.m_pad // tile_size[0]
        self.n_split = self.n_pad // tile_size[1]
        self.slices = self.create_slices()

    def create_slices(self):
        slices = []
        for i in range(self.m_split):
            for j in range(self.n_split):
                addrA = TileAddress(
                    (0, i * self.tile_size[0]), 
                    (self.k, self.tile_size[0]), 
                    self.annotations['A']
                )
                addrB = TileAddress(
                    (j * self.tile_size[1], 0), 
                    (self.tile_size[1], self.k), 
                    self.annotations['B']
                )
                slice = SliceMatmul(
                    m=self.tile_size[0],
                    k=self.k,
                    n=self.tile_size[1],
                    addrA=addrA,
                    addrB=addrB,
                    annotationRes=self.annotations['Res'])
                slices.append(slice)
        return slices

class SliceSoftmax:
    def __init__(self, seqnum, addrIn, addrRes):
        self.seqnum = seqnum
        self.addrIn = addrIn # TileAddress
        self.addrRes = addrRes # TileAddress
        self.complete_event = None
        self.tile_id = None

    def __str__(self):
        return f"SliceSoftmax(seqnum={self.seqnum}, addrIn={self.addrIn}, addrRes={self.addrRes})"

    def __repr__(self):
        return f"SliceSoftmax(seqnum={self.seqnum}, addrIn={self.addrIn}, addrRes={self.addrRes})"

class Softmax:
    def __init__(self, seqlen):
        self.seqlen = seqlen
        self.slices = self.create_slices()

    def create_slices(self):
        slices = []
        for i in range(self.seqlen):
            addrIn = TileAddress(
                (0, i), 
                (self.seqlen, 1), 
                'QK'
            )
            addrRes = TileAddress(
                (0, i), 
                (self.seqlen, 1), 
                'Softmax'
            )
            slice = SliceSoftmax(seqnum=i, addrIn=addrIn, addrRes=addrRes)
            slices.append(slice)
        return slices
        

class Memory(simpy.Container):
    def __init__(self, env, size, init=0):
        super().__init__(env, capacity=size, init=init)
        self.allocate_list = [] # list of TileAddress objects
        self.size = size

    def allocate(self, addr):
        self.get(addr.offset[0] * addr.offset[1])
        self.allocate_list.append(addr)

    def deallocate(self, amount):
        yield self.put(amount)


class RISCVTile():
    def __init__(self, env, tile_id):
        self.tile_id = tile_id
        self.compute_unit = simpy.Resource(env, capacity=1)
        self.dma_unit = simpy.Resource(env, capacity=1)
        self.noc_unit = simpy.Resource(env, capacity=1)
        self.local_memory = Memory(env, LOCAL_MEMORY_SIZE, init=0)
        self.task_num = 0

    def softmax(self, data):
        pass

    def matmul(self, data):
        pass

class RISCVMultiprocessor:
    def __init__(self, env, num_cores, compute_graph):
        self.env = env
        # self.cores = simpy.Resource(env, num_cores)
        self.cores = [RISCVTile(env, i) for i in range(num_cores)]
        self.global_memory = Memory(env, GLOBAL_MEMORY_SIZE, init=0)
        self.compute_graph = compute_graph

    def search_parent(self, node):
        parents = list(self.compute_graph.predecessors(node))
        return parents
    
    def search_overlap(self, addr):
        overlap_list = []
        for core in self.cores:
            for allocated_addr in core.local_memory.allocate_list:
                overlap_addr = overlap(allocated_addr, addr)
                if overlap_addr:
                    overlap_list.append((core.tile_id, overlap_addr))
        return overlap_list

    def analyze_workload(self, workload):
        parents = self.search_parent(workload)
        memory_trace = []
        noc_trace = []
        if not parents:
            memory_trace.append(('global', workload.addrA))
            memory_trace.append(('global', workload.addrB))
        else:
            for parent in parents:
                noc_trace.append(parent)
        return memory_trace, noc_trace

    def schedule(self, noc_trace):
        core_idx = 0
        # Q, K, V projections
        if len(noc_trace) == 0:
            min_task_num = float('inf')
            for i, core in enumerate(self.cores):
                if core.task_num < min_task_num:
                    min_task_num = core.task_num
                    core_idx = i
        else:
            core_size_map = {}
            for trace in noc_trace:
                if trace.tile_id is not None:
                    core_size_map[trace.tile_id] = core_size_map.get(trace.tile_id, 0) + trace.addrRes.offset[0] * trace.addrRes.offset[1]
            max_size = 0
            for i, core in enumerate(self.cores):
                if core.tile_id in core_size_map:
                    size = core_size_map[core.tile_id]
                else:
                    size = 0
                if size > max_size:
                    max_size = size
                    core_idx = i
                elif size == max_size:
                    if core.task_num < self.cores[core_idx].task_num:
                        core_idx = i
        return core_idx
    
    def execute(self, workload): # workload is a slice of a Matmul or Softmax
        workload.complete_event = self.env.event()
        # Step 0: Analyze the workload

        memory_trace, noc_trace = self.analyze_workload(workload)
        for trace in noc_trace:
            yield trace.complete_event

        core_idx = self.schedule(noc_trace)
        self.cores[core_idx].task_num += 1
        
        for trace in memory_trace:

            with self.cores[core_idx].dma_unit.request() as request:
                yield request
                if trace[1] in self.cores[core_idx].local_memory.allocate_list:
                    # print(f"Skipping allocation of {trace[1]} on core {core_idx} at time {self.env.now} as it is already allocated")
                    continue
                yield self.env.timeout(50)
                # print(f"Allocating {trace[1]} on core {core_idx} at time {self.env.now}")
                self.cores[core_idx].local_memory.allocate(trace[1])

        for trace in noc_trace:
            if trace.tile_id == core_idx:
                continue
            with self.cores[core_idx].noc_unit.request() as request:
                yield request
                if trace.addrRes in self.cores[core_idx].local_memory.allocate_list:
                    # print(f"Skipping transfer of {trace.addrRes} from core {trace.tile_id} to core {core_idx} at time {self.env.now} as it is already allocated")
                    continue
                yield self.env.timeout(20)
                # print(f"Transferring {trace.addrRes} from core {trace.tile_id} to core {core_idx} at time {self.env.now}")
                self.cores[core_idx].local_memory.allocate(trace.addrRes)

        with self.cores[core_idx].compute_unit.request() as request:
            workload.state = 'executing'
            yield request
            yield self.env.timeout(5)
            workload.state = 'completed'
            workload.tile_id = core_idx
            self.cores[core_idx].local_memory.allocate(workload.addrRes)
            # print(f"Workload {workload} executed on core {core_idx} with tile_id {workload.tile_id} at time {self.env.now}")
        self.cores[core_idx].task_num -= 1
        workload.complete_event.succeed()

RANDOM_SEED = 42



class Attention:
    def __init__(self, seqlen, dim, config):
        self.seqlen = seqlen
        self.dim = dim

        # Project Q, K, V
        self.proj_q = Matmul(
            m=seqlen, 
            k=dim, 
            n=dim, 
            annotations={'A': 'X', 'B': 'Wq', 'Res': 'Q'}, 
            # tile_size=(16, 16)
            tile_size=config.get('proj_q_tile_size', (16, 16))
        )
        self.proj_k = Matmul(
            m=seqlen, 
            k=dim, 
            n=dim, 
            annotations={'A': 'X', 'B': 'Wk', 'Res': 'K'}, 
            tile_size=config.get('proj_k_tile_size', (16, 16))
        )
        self.proj_v = Matmul(
            m=seqlen, 
            k=dim, 
            n=dim, 
            annotations={'A': 'X', 'B': 'Wv', 'Res': 'V'}, 
            tile_size=config.get('proj_v_tile_size', (16, 16))
        )

        # QK Matmul
        self.qk_matmul = Matmul(
            m=seqlen, 
            k=dim, 
            n=seqlen, 
            annotations={'A': 'Q', 'B': 'K', 'Res': 'QK'}, 
            tile_size=config.get('qk_matmul_tile_size', (16, 16))
        )

        # Softmax
        self.qk_softmax = Softmax(seqlen)

        # QKV Matmul
        self.qkv_matmul = Matmul(
            m=seqlen, 
            k=seqlen, 
            n=dim, 
            annotations={'A': 'Softmax', 'B': 'V', 'Res': 'Output'}, 
            tile_size=config.get('qkv_matmul_tile_size', (16, 16))
        )
        self.graph = self.create_graph()

    def create_graph(self):
        graph = nx.DiGraph()
        # add q, k, v projections
        def add_node(slice):
            if slice not in graph:
                graph.add_node(slice, label=slice.addrRes.annotation)

        def try_add_edge(slice_src, slice_dest):
            add_node(slice_src)
            add_node(slice_dest)
            graph.add_edge(slice_src, slice_dest)
        
        for slice in self.proj_q.slices:
            add_node(slice)
        for slice in self.proj_k.slices:
            add_node(slice)
        for slice in self.proj_v.slices:
            add_node(slice)

        for slice in self.qk_matmul.slices:
            nodes = list(graph.nodes)
            for node in nodes:
                if is_overlap(node.addrRes, slice.addrA):
                    try_add_edge(node, slice)
                if is_overlap(node.addrRes, slice.addrB):
                    try_add_edge(node, slice)

        for slice in self.qk_softmax.slices:
            nodes = list(graph.nodes)
            for node in nodes:
                if is_overlap(node.addrRes, slice.addrIn):
                    try_add_edge(node, slice)

        for slice in self.qkv_matmul.slices:
            nodes = list(graph.nodes)
            for node in nodes:
                if is_overlap(node.addrRes, slice.addrA):
                    try_add_edge(node, slice)
                if is_overlap(node.addrRes, slice.addrB):
                    try_add_edge(node, slice)
        return graph

# def annotation_to_pos(node):
#     annotation = node.addrRes.annotation
#     base = node.addrRes.base + node.addrRes.offset
#     if annotation == 'X':
#         return (base[0] + 0, base[1] + 0)
#     elif annotation == 'Wq':
#         return (base[0] + 0, base[1] + 100)
#     elif annotation == 'Wk':
#         return (base[0] + 0, base[1] + 200)
#     elif annotation == 'Wv':
#         return (base[0] + 0, base[1] + 300)
#     elif annotation == 'Q':
#         return (base[0] + 100, base[1] + 100)
#     elif annotation == 'K':
#         return (base[0] + 100, base[1] + 200)
#     elif annotation == 'V':
#         return (base[0] + 100, base[1] + 300)
#     elif annotation == 'QK':
#         return (base[0] + 300, base[1] + 100)
#     elif annotation == 'Softmax':
#         return (base[0] + 500, base[1] + 100)
#     elif annotation == 'Output':
#         return (base[0] + 700, base[1] + 100)

import time

SEQLEN = 512
DIM = 512

def evaluation(config):
    start = time.time()
    env = simpy.Environment()
    attn = Attention(seqlen=SEQLEN, dim=DIM, config=config)
    processor = RISCVMultiprocessor(env, num_cores=config['num_cores'], compute_graph=attn.graph)
    topological_sort = list(nx.topological_sort(attn.graph))
    for node in topological_sort:
        env.process(processor.execute(node))
    env.run()
    end = time.time()
    return env.now, end - start

def main():
    tile_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (80, 80), (96, 96), (112, 112), (128, 128)]
    num_cores = 8
    simulation_times = []
    for tile_size in tile_sizes:
        config = {
            'num_cores': num_cores,
            'proj_q_tile_size': tile_size,
            'proj_k_tile_size': tile_size,
            'proj_v_tile_size': tile_size,
            'qk_matmul_tile_size': tile_size,
            'qkv_matmul_tile_size': tile_size,
        }
        print(f"Running simulation with tile size {tile_size} and {num_cores} cores...")
        simulation_time, _ = evaluation(config)
        simulation_times.append((tile_size, simulation_time))

    # draw line chart
    plt.figure(figsize=(10, 6))
    plt.plot([size[0][0] for size in simulation_times], [size[1] for size in simulation_times], marker='o')
    plt.title('Simulation Time vs Tile Size')
    plt.xlabel('Tile Size (x, y)')
    plt.ylabel('Simulation Time (seconds)')
    plt.xticks([size[0][0] for size in simulation_times])
    plt.grid()
    # plt.show()
    plt.savefig('simulation_time_vs_tile_size.png')

    tile_size = [(64, 64)]
    num_cores = [1, 2, 4, 8, 16, 32, 64, 128]
    simulation_times = []
    for num_core in num_cores:
        config = {
            'num_cores': num_core,
            'proj_q_tile_size': tile_size[0],
            'proj_k_tile_size': tile_size[0],
            'proj_v_tile_size': tile_size[0],
            'qk_matmul_tile_size': tile_size[0],
            'qkv_matmul_tile_size': tile_size[0],
        }
        print(f"Running simulation with tile size {tile_size[0]} and {num_core} cores...")
        simulation_time, _ = evaluation(config)
        simulation_times.append((num_core, simulation_time))

    # draw histogram
    plt.figure(figsize=(10, 6))
    plt.bar([str(size[0]) for size in simulation_times], [size[1] for size in simulation_times])
    plt.title('Simulation Time vs Number of Cores')
    plt.xlabel('Number of Cores')
    plt.ylabel('Simulation Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    # plt.show()
    plt.savefig('simulation_time_vs_num_cores.png')


    # start = time.time()
    # config = {
    #     'num_cores': 64,
    #     'proj_q_tile_size': (64, 64),
    #     'proj_k_tile_size': (64, 64),
    #     'proj_v_tile_size': (64, 64),
    #     'qk_matmul_tile_size': (64, 64),
    #     'qkv_matmul_tile_size': (64, 64),
    # }
    # env = simpy.Environment()
    # attn = Attention(seqlen=SEQLEN, dim=DIM, config=config)
    # processor = RISCVMultiprocessor(env, num_cores=config['num_cores'], compute_graph=attn.graph)
    # topological_sort = list(nx.topological_sort(attn.graph))
    # for node in topological_sort:
    #     env.process(processor.execute(node))
    # env.run()
    # end = time.time()
    # print("Simulation completed.")
    # print('Time taken:', env.now)
    # print('Simulation time:', end - start)

if __name__ == "__main__":
    main()
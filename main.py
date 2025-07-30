import simpy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# import random

RANDOM_SEED = 42
LOCAL_MEMORY_SIZE = 4096
GLOBAL_MEMORY_SIZE = 16384

gemm_table = pd.read_csv('gemm_4cores_0729_filtered.csv')
softmax_table = pd.read_csv('softmax_4.csv')

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

class MemoryTrace:
    def __init__(self, start_time, end_time, core_idx, type, addr):
        self.start_time = start_time
        self.end_time = end_time
        self.core_idx = core_idx
        self.type = type # e.g. 'load', 'store'
        self.addr = addr # TileAddress

class NoCTrace:
    def __init__(self, start_time, end_time, src_core_idx, dest_core_idx, type, addr):
        self.start_time = start_time
        self.end_time = end_time
        self.src_core_idx = src_core_idx
        self.dest_core_idx = dest_core_idx
        self.type = type # e.g. 'unicast'
        self.addr = addr # TileAddress

class ComputeTrace:
    def __init__(self, start_time, end_time, core_idx, workload):
        self.start_time = start_time
        self.end_time = end_time
        self.core_idx = core_idx
        self.workload = workload

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

memory_record = []
compute_record = []
noc_record = []

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
                # memory_req_trace.append(MemoryRequest(self.env.now, core_idx, 'load', trace[1]))
                start_time = self.env.now
                yield self.env.timeout(trace[1].offset[0]*trace[1].offset[1]/1e7)
                memory_record.append(MemoryTrace(start_time, self.env.now, core_idx, 'load', trace[1]))
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
                start_time = self.env.now
                yield self.env.timeout(trace.addrRes.offset[0]*trace.addrRes.offset[1]/1e8)
                noc_record.append(NoCTrace(start_time, self.env.now, trace.tile_id, core_idx, 'unicast', trace.addrRes))
                # print(f"Transferring {trace.addrRes} from core {trace.tile_id} to core {core_idx} at time {self.env.now}")
                self.cores[core_idx].local_memory.allocate(trace.addrRes)

        with self.cores[core_idx].compute_unit.request() as request:
            workload.state = 'executing'
            yield request
            start_time = self.env.now
            if isinstance(workload, SliceMatmul):
                compute_time = gemm_table[(gemm_table['M'] == workload.addrA.offset[1]) &
                                          (gemm_table['N'] == workload.addrB.offset[0]) &
                                          (gemm_table['K'] == workload.addrA.offset[0])]['op_time'].values[0]
            elif isinstance(workload, SliceSoftmax):
                compute_time = softmax_table[softmax_table['input_size'] == workload.addrIn.offset[0]]['time(nsec)'].values[0]/ 1e9
            yield self.env.timeout(compute_time)
            compute_record.append(ComputeTrace(start_time, self.env.now, core_idx, workload))
            workload.state = 'completed'
            workload.tile_id = core_idx
            self.cores[core_idx].local_memory.allocate(workload.addrRes)
        self.cores[core_idx].task_num -= 1
        workload.complete_event.succeed()
        if workload.addrRes.annotation == 'Output':
            with self.cores[core_idx].dma_unit.request() as request:
                yield request
                start_time = self.env.now
                yield self.env.timeout(workload.addrRes.offset[0]*workload.addrRes.offset[1]/1e7)
                memory_record.append(MemoryTrace(start_time, self.env.now, core_idx, 'store', workload.addrRes))

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
    tile_size = (64, 64)  # Example tile size, can be adjusted
    num_cores = 8
    config = {
        'num_cores': num_cores,
        'proj_q_tile_size': tile_size,
        'proj_k_tile_size': tile_size,
        'proj_v_tile_size': tile_size,
        'qk_matmul_tile_size': tile_size,
        'qkv_matmul_tile_size': tile_size,
    }
    print(f"Running simulation with tile size {tile_size} and {num_cores} cores...")
    simulation_time, elapsed_time = evaluation(config)
    print(f"Simulation time: {simulation_time}, Elapsed time: {elapsed_time}")

# Draw gantt graph
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111)
    # for core in range(num_cores):
    #     core_compute_trace = [trace for trace in compute_record if trace.core_idx == core]
    #     for trace in core_compute_trace:
    #         ax.barh(core, trace.end_time - trace.start_time, left=trace.start_time, label=f'Core {core} - {trace.workload}')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Core')
    # ax.set_title('Gantt Chart of Core Execution')
    # ax.set_yticks(range(num_cores))
    # ax.set_yticklabels([f'Core {i}' for i in range(num_cores)])
    # # ax.legend()
    # plt.tight_layout()
    # plt.savefig('gantt_chart.png')

    # Draw gantt graph for core 0's compute, memory, and NoC traces in separate rows
    for core_idx in range(1):
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        core_compute_trace = [trace for trace in compute_record if trace.core_idx == core_idx]
        core_memory_trace = [trace for trace in memory_record if trace.core_idx == core_idx]
        core_noc_trace = [trace for trace in noc_record if trace.dest_core_idx == core_idx]
        for trace in core_compute_trace:
            if trace.workload.addrRes.annotation == 'Output':
                ax.barh(0, trace.end_time - trace.start_time, left=trace.start_time, color='blue', label='Softmax(QK^T)*V' if trace == core_compute_trace[0] else "")
            elif trace.workload.addrRes.annotation == 'QK':
                ax.barh(0, trace.end_time - trace.start_time, left=trace.start_time, color='red', label='QK^T' if trace == core_compute_trace[0] else "")
            elif trace.workload.addrRes.annotation == 'Softmax':
                ax.barh(0, trace.end_time - trace.start_time, left=trace.start_time, color='purple', label='Softmax' if trace == core_compute_trace[0] else "")
            elif trace.workload.addrRes.annotation == 'Q':
                ax.barh(0, trace.end_time - trace.start_time, left=trace.start_time, color='darkcyan', label='Q Projection' if trace == core_compute_trace[0] else "")
            elif trace.workload.addrRes.annotation == 'K':
                ax.barh(0, trace.end_time - trace.start_time, left=trace.start_time, color='darkmagenta', label='K Projection' if trace == core_compute_trace[0] else "")
            elif trace.workload.addrRes.annotation == 'V':
                ax.barh(0, trace.end_time - trace.start_time, left=trace.start_time, color='darkviolet', label='V Projection' if trace == core_compute_trace[0] else "")
        for trace in core_memory_trace:
            ax.barh(1, trace.end_time - trace.start_time, left=trace.start_time, color='green', label='Memory' if trace == core_memory_trace[0] else "")
        for trace in core_noc_trace:
            ax.barh(2, trace.end_time - trace.start_time, left=trace.start_time, color='orange', label='NoC' if trace == core_noc_trace[0] else "")
        ax.set_xlabel('Time')
        # ax.set_xlim(left=0, right=4000)
        ax.set_ylabel('Core')
        ax.set_title(f'Gantt Chart of Core {core_idx} Execution')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([f'Core {core_idx} - Compute', f'Core {core_idx} - Memory', f'Core {core_idx} - NoC'])
        ax.legend(handles=[plt.Line2D([0], [0], color='blue', lw=4),
                           plt.Line2D([0], [0], color='red', lw=4),
                           plt.Line2D([0], [0], color='purple', lw=4),
                           plt.Line2D([0], [0], color='darkcyan', lw=4),
                           plt.Line2D([0], [0], color='darkmagenta', lw=4),
                           plt.Line2D([0], [0], color='darkviolet', lw=4),
                           plt.Line2D([0], [0], color='green', lw=4),
                           plt.Line2D([0], [0], color='orange', lw=4)],
                  labels=['Softmax(QK^T)*V', 'QK^T', 'Softmax', 'Q Projection', 'K Projection', 'V Projection', 'Memory', 'NoC'], loc='upper left')
        plt.tight_layout()
        plt.savefig(f'gantt_chart_core_{core_idx}.png')

    # print(f"Memory Requests: {len(memory_req_trace)}")
    # print(f"NoC Requests: {len(noc_req_trace)}")
    # # Dump memory and NoC request traces
    # with open('memory_trace.csv', 'w') as f:
    #     f.write('time,core_idx,type,data,base1,base2,offset1,offset2\n')
    #     for req in memory_req_trace:
    #         f.write(f"{req.time},{req.core_idx},{req.type},{req.addr.annotation},{req.addr.base[0]},{req.addr.base[1]},{req.addr.offset[0]},{req.addr.offset[1]}\n")
    # with open('noc_trace.csv', 'w') as f:
    #     f.write('time,src_core_idx,dest_core_idx,type,data,base1,base2,offset1,offset2\n')
    #     for req in noc_req_trace:
    #         f.write(f"{req.time},{req.src_core_idx},{req.dest_core_idx},{req.type},{req.addr.annotation},{req.addr.base[0]},{req.addr.base[1]},{req.addr.offset[0]},{req.addr.offset[1]}\n")
    # tile_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (80, 80), (96, 96), (112, 112), (128, 128)]
    # num_cores = 8
    # simulation_times = []
    # for tile_size in tile_sizes:
    #     config = {
    #         'num_cores': num_cores,
    #         'proj_q_tile_size': tile_size,
    #         'proj_k_tile_size': tile_size,
    #         'proj_v_tile_size': tile_size,
    #         'qk_matmul_tile_size': tile_size,
    #         'qkv_matmul_tile_size': tile_size,
    #     }
    #     print(f"Running simulation with tile size {tile_size} and {num_cores} cores...")
    #     simulation_time, _ = evaluation(config)
    #     simulation_times.append((tile_size, simulation_time))

    # # draw line chart
    # plt.figure(figsize=(10, 6))
    # plt.plot([size[0][0] for size in simulation_times], [size[1] for size in simulation_times], marker='o')
    # plt.title('Simulation Time vs Tile Size')
    # plt.xlabel('Tile Size (x, y)')
    # plt.ylabel('Simulation Time (seconds)')
    # plt.xticks([size[0][0] for size in simulation_times])
    # plt.grid()
    # # plt.show()
    # plt.savefig('simulation_time_vs_tile_size.png')

    # tile_size = [(64, 64)]
    # num_cores = [1, 2, 4, 8, 16, 32, 64, 128]
    # simulation_times = []
    # for num_core in num_cores:
    #     config = {
    #         'num_cores': num_core,
    #         'proj_q_tile_size': tile_size[0],
    #         'proj_k_tile_size': tile_size[0],
    #         'proj_v_tile_size': tile_size[0],
    #         'qk_matmul_tile_size': tile_size[0],
    #         'qkv_matmul_tile_size': tile_size[0],
    #     }
    #     print(f"Running simulation with tile size {tile_size[0]} and {num_core} cores...")
    #     simulation_time, _ = evaluation(config)
    #     simulation_times.append((num_core, simulation_time))

    # # draw histogram
    # plt.figure(figsize=(10, 6))
    # plt.bar([str(size[0]) for size in simulation_times], [size[1] for size in simulation_times])
    # plt.title('Simulation Time vs Number of Cores')
    # plt.xlabel('Number of Cores')
    # plt.ylabel('Simulation Time (seconds)')
    # plt.xticks(rotation=45)
    # plt.grid(axis='y')
    # # plt.show()
    # plt.savefig('simulation_time_vs_num_cores.png')


if __name__ == "__main__":
    main()
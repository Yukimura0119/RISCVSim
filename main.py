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

    def __str__(self):
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

    def __str__(self):
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

    def softmax(self, data):
        pass

    def matmul(self, data):
        pass

class RISCVMultiprocessor:
    def __init__(self, env, num_cores):
        self.env = env
        # self.cores = simpy.Resource(env, num_cores)
        self.cores = [RISCVTile(env, i) for i in range(num_cores)]
        self.global_memory = Memory(env, GLOBAL_MEMORY_SIZE, init=0)

    def search_overlap(self, addr):
        overlap_list = []
        # if global_memory:
        #     for allocated_addr in self.global_memory.allocate_list:
        #         overlap_addr = overlap(allocated_addr, addr)
        #         if overlap_addr:
        #             overlap_list.append(('global', overlap_addr))
        #     return overlap_list
        for core in self.cores:
            for allocated_addr in core.local_memory.allocate_list:
                overlap_addr = overlap(allocated_addr, addr)
                if overlap_addr:
                    overlap_list.append((core.tile_id, overlap_addr))
        return overlap_list 


    def analyze_workload(self, workload):
        # return a list of memory queries.
        global_list = ['X', 'Wq', 'Wk', 'Wv']
        memory_req = []
        noc_req = []
        if isinstance(workload, SliceMatmul):
            if workload.addrA.annotation in global_list:
                memory_req.append(('global', workload.addrA))
            else:
                overlap_listA = self.search_overlap(workload.addrA)
                if overlap_listA:
                    noc_req.extend(overlap_listA)

            if workload.addrB.annotation in global_list:
                memory_req.append(('global', workload.addrB))
            else:
                overlap_listB = self.search_overlap(workload.addrB)
                if overlap_listB:
                    noc_req.extend(overlap_listB)
        elif isinstance(workload, SliceSoftmax):
            overlap_listIn = self.search_overlap(workload.addrIn)
            if overlap_listIn:
                noc_req.extend(overlap_listIn)
        return memory_req, noc_req

    def execute(self, workload): # workload is a slice of a Matmul or Softmax
        # Step 0: Analyze the workload
        memory_req, noc_req = self.analyze_workload(workload)
        # remove duplicates
        memory_req = list(set(memory_req))
        noc_req = list(set(noc_req))
        # Step 1: schedule the workload to a core
        core_idx = schedule()
        for mem_req in memory_req:
            self.cores[core_idx].local_memory.allocate(mem_req[1])
            with self.cores[core_idx].dma_unit.request() as request:
                yield request
                print(f"Core {core_idx} accessing memory for {mem_req[1]} at time {self.env.now}")
                yield self.env.timeout(1)  # Simulate DMA transfer time
        
        for noc_req in noc_req:
            self.cores[core_idx].local_memory.allocate(noc_req[1])
            with self.cores[core_idx].noc_unit.request() as request:
                yield request
                print(f"Core {core_idx} sending data to NoC for {noc_req[1]} at time {self.env.now}")
                yield self.env.timeout(1)  # Simulate NoC transfer time

        # compute unit execution
        self.cores[core_idx].local_memory.allocate(workload.addrRes)
        with self.cores[core_idx].compute_unit.request() as request:
            yield request
            print(f"Core {core_idx} executing workload {workload} at time {self.env.now}")
            yield self.env.timeout(1)

def schedule():
    core_idx = 0
    return core_idx

RANDOM_SEED = 42

class Attention:
    def __init__(self, seqlen, dim):
        self.seqlen = seqlen
        self.dim = dim

        # Project Q, K, V
        self.proj_q = Matmul(
            m=seqlen, 
            k=dim, 
            n=dim, 
            annotations={'A': 'X', 'B': 'Wq', 'Res': 'Q'}, 
            tile_size=(16, 16)
        )
        self.proj_k = Matmul(
            m=seqlen, 
            k=dim, 
            n=dim, 
            annotations={'A': 'X', 'B': 'Wk', 'Res': 'K'}, 
            tile_size=(16, 16)
        )
        self.proj_v = Matmul(
            m=seqlen, 
            k=dim, 
            n=dim, 
            annotations={'A': 'X', 'B': 'Wv', 'Res': 'V'}, 
            tile_size=(16, 16)
        )

        # QK Matmul
        self.qk_matmul = Matmul(
            m=seqlen, 
            k=dim, 
            n=seqlen, 
            annotations={'A': 'Q', 'B': 'K', 'Res': 'QK'}, 
            tile_size=(16, 16)
        )

        # Softmax
        self.qk_softmax = Softmax(seqlen)

        # QKV Matmul
        self.qkv_matmul = Matmul(
            m=seqlen, 
            k=seqlen, 
            n=dim, 
            annotations={'A': 'Softmax', 'B': 'V', 'Res': 'Output'}, 
            tile_size=(16, 16)
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

def main():
    # TileA = TileAddress((15, 17), (16, 16), 'X')
    # TileB = TileAddress((15, 15), (16, 16), 'X')
    # print(overlap(TileA, TileB))  # Should return True
    env = simpy.Environment()
    processor = RISCVMultiprocessor(env, num_cores=1)
    attn = Attention(seqlen=64, dim=64)
    topological_sort = list(nx.topological_sort(attn.graph))
    for node in topological_sort:
        # print(f"Processing node: {node}")
        env.process(processor.execute(node))
    env.run(until=1000)
    # Draw the graph
    # pos = {
    #     node: annotation_to_pos(node) for node in attn.graph.nodes()
    # }
    # labels = {
    #     node: node.addrRes.annotation for node in attn.graph.nodes()
    # }
    # plt.figure(figsize=(8, 6))
    # nx.draw(
    #     attn.graph, pos,
    #     with_labels=True,
    #     labels=labels,
    #     node_size=1000,
    #     node_color='lightgreen',
    #     font_size=10,
    #     font_weight='bold',
    #     arrows=True,
    #     arrowsize=20
    # )
    # plt.title("DAG with Custom Node Objects")
    # plt.axis("equal")
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
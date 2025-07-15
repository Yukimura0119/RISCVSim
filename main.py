import simpy
import networkx as nx
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
    
    def __hash__(self):
        return hash((self.base, self.offset, self.annotation))
    
    def __eq__(self, other):
        if not isinstance(other, TileAddress):
            return False
        return (self.base == other.base and 
                self.offset == other.offset and 
                self.annotation == other.annotation)

def overlap(addrA, addrB):
    return (addrA.base[0] < addrB.base[0] + addrB.offset[0] and
            addrA.base[0] + addrA.offset[0] > addrB.base[0] and
            addrA.base[1] < addrB.base[1] + addrB.offset[1] and
            addrA.base[1] + addrA.offset[1] > addrB.base[1] and 
            addrA.annotation == addrB.annotation)

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
    def __init__(self, seqnum, addrIn, addrOut):
        self.seqnum = seqnum
        self.addrIn = addrIn # TileAddress
        self.addrOut = addrOut # TileAddress

    def __str__(self):
        return f"SliceSoftmax(seqnum={self.seqnum}, addrIn={self.addrIn}, addrOut={self.addrOut})"

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
            addrOut = TileAddress(
                (0, i), 
                (self.seqlen, 1), 
                'Softmax'
            )
            slice = SliceSoftmax(seqnum=i, addrIn=addrIn, addrOut=addrOut)
            slices.append(slice)
        return slices
        

class Memory(simpy.Container):
    def __init__(self, env, size, init=0):
        super().__init__(env, capacity=size, init=init)
        self.allocate_list = [] # list of TileAddress objects
        self.size = size

    def allocate(self, amount):
        if amount > self.size:
            pass
        yield self.get(amount)

    def deallocate(self, amount):
        yield self.put(amount)


class RISCVTile(simpy.Resource):
    def __init__(self, env, tile_id):
        super().__init__(env)
        self.tile_id = tile_id
        self.local_memory = simpy.Container(env, LOCAL_MEMORY_SIZE, init=LOCAL_MEMORY_SIZE)

    def softmax(self, data):
        pass

    def matmul(self, data):
        pass

    def execute(self, workload):
        yield self.env.timeout(5)
        pass

class RISCVMultiprocessor:
    def __init__(self, env, num_cores):
        self.env = env
        # self.cores = simpy.Resource(env, num_cores)
        self.cores = [RISCVTile(env, i) for i in range(num_cores)]
        self.global_memory = simpy.Container(env, GLOBAL_MEMORY_SIZE, init=GLOBAL_MEMORY_SIZE)

    def softmax(self, data):
        pass

    def matmul(self, data):
        pass

    def execute(self, workload):
        yield self.env.timeout(5)
        pass

def schedule(env, processor, type):
    core_idx = 3
    return core_idx

def workload(env, processor, type):
    # schedule
    core_idx = schedule(env, processor, type)
    core = processor.cores[core_idx]
    with core.request() as request:
        yield request
        print(f"Core {core.tile_id} executing workload of type {type} at time {env.now}")
        yield env.process(core.execute(type))


RANDOM_SEED = 42

def merge_graphs(graphA, graphB):
    merged_graph = nx.DiGraph()
    merged_graph.add_nodes_from(graphA.nodes(data=True))
    merged_graph.add_edges_from(graphA.edges(data=True))
    return merged_graph

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
        def add_node(addr):
            if addr not in graph:
                graph.add_node(addr, label=addr.annotation)

        def try_add_edge(from_addr, to_addr):
            add_node(from_addr)
            add_node(to_addr)
            graph.add_edge(from_addr, to_addr)
        
        for slice in self.proj_q.slices:
            try_add_edge(slice.addrA, slice.addrRes)
            try_add_edge(slice.addrB, slice.addrRes)
        for slice in self.proj_k.slices:
            try_add_edge(slice.addrA, slice.addrRes)
            try_add_edge(slice.addrB, slice.addrRes)
        for slice in self.proj_v.slices:
            try_add_edge(slice.addrA, slice.addrRes)
            try_add_edge(slice.addrB, slice.addrRes)

        for slice in self.qk_matmul.slices:
            nodes = list(graph.nodes)
            for node in nodes:
                if isinstance(node, TileAddress):
                    if overlap(node, slice.addrA):
                        try_add_edge(node, slice.addrRes)
                    if overlap(node, slice.addrB):
                        try_add_edge(node, slice.addrRes)

        for slice in self.qk_softmax.slices:
            nodes = list(graph.nodes)
            for node in nodes:
                if isinstance(node, TileAddress):
                    if overlap(node, slice.addrIn):
                        try_add_edge(node, slice.addrOut)

        for slice in self.qkv_matmul.slices:
            nodes = list(graph.nodes)
            for node in nodes:
                if isinstance(node, TileAddress):
                    if overlap(node, slice.addrA):
                        try_add_edge(node, slice.addrRes)
                    if overlap(node, slice.addrB):
                        try_add_edge(node, slice.addrRes)
        return graph

def annotation_to_pos(node):
    annotation = node.annotation
    base = node.base + node.offset
    if annotation == 'X':
        return (base[0] + 0, base[1] + 0)
    elif annotation == 'Wq':
        return (base[0] + 0, base[1] + 100)
    elif annotation == 'Wk':
        return (base[0] + 0, base[1] + 200)
    elif annotation == 'Wv':
        return (base[0] + 0, base[1] + 300)
    elif annotation == 'Q':
        return (base[0] + 100, base[1] + 100)
    elif annotation == 'K':
        return (base[0] + 100, base[1] + 200)
    elif annotation == 'V':
        return (base[0] + 100, base[1] + 300)
    elif annotation == 'QK':
        return (base[0] + 300, base[1] + 100)
    elif annotation == 'Softmax':
        return (base[0] + 500, base[1] + 100)
    elif annotation == 'Output':
        return (base[0] + 700, base[1] + 100)

import matplotlib.pyplot as plt
def main():
    attn = Attention(seqlen=64, dim=64)
    pos = {
        # X -> +100
        # Wq -> +200
        # node: node.base for node in attn.graph.nodes()
        node: annotation_to_pos(node) for node in attn.graph.nodes()
    }
    labels = {
        node: node.annotation for node in attn.graph.nodes()
    }
    plt.figure(figsize=(8, 6))
    nx.draw(
        attn.graph, pos,
        with_labels=True,
        labels=labels,
        node_size=1000,
        node_color='lightgreen',
        font_size=10,
        font_weight='bold',
        arrows=True,
        arrowsize=20
    )
    plt.title("DAG with Custom Node Objects")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
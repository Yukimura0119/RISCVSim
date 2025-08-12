import simpy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import subprocess
import time
# import random

LOCAL_MEMORY_SIZE = 4096
GLOBAL_MEMORY_SIZE = 16384
SEQLEN = 512
DIM = 512
RANDOM_SEED = 42

def round_decimal(x, digits=3):
    factor = 10 ** digits
    return round(x * factor) / factor

def truncate_decimal(x, digits=3):
    factor = 10 ** digits
    return int(x * factor) / factor

TABLE_PATH = './compute_tables/'
gemm_tables = [pd.read_csv(TABLE_PATH + 'gemm_1Cores_0802.csv'), pd.read_csv(TABLE_PATH + 'gemm_2Cores_0802.csv'), 
               pd.read_csv(TABLE_PATH + 'gemm_4Cores_0802.csv')]
softmax_tables = [pd.read_csv(TABLE_PATH + 'softmax_compute_1.csv'), pd.read_csv(TABLE_PATH + 'softmax_compute_2.csv'),
                  pd.read_csv(TABLE_PATH + 'softmax_compute_4.csv')]

def get_gemm_compute_time_api(m, n, k, num_cores):
    cmd = f'printf "gemm {m} {n} {k}" | nc -q1 192.168.1.115 8000'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if num_cores == 1:
        compute_time = float(lines[0].split()[1]) * 1e6
    elif num_cores == 2:
        compute_time = float(lines[1].split()[1]) * 1e6
    elif num_cores == 4:
        compute_time = float(lines[2].split()[1]) * 1e6
    else:
        raise ValueError("Unsupported number of cores")
    return compute_time

def get_softmax_compute_time_api(input_size, num_cores):
    cmd = f'printf "softmax {input_size}" | nc -q1 192.168.1.115 8000'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if num_cores == 1:
        compute_time = float(lines[0].split()[1]) * 1e6
    elif num_cores == 2:
        compute_time = float(lines[1].split()[1]) * 1e6
    elif num_cores == 4:
        compute_time = float(lines[2].split()[1]) * 1e6
    else:
        raise ValueError("Unsupported number of cores")
    return compute_time

def get_gemm_compute_time(m, n, k, num_cores):
    if num_cores == 1:
        table = gemm_tables[0]
    elif num_cores == 2:
        table = gemm_tables[1]
    elif num_cores == 4:
        table = gemm_tables[2]
    else:
        raise ValueError("Unsupported number of cores")
    compute_time = table[(table['Mc'] == m) & (table['Nc'] == n) & (table['Kc'] == k)]['op_time'].values[0] * 1e6
    return compute_time

def get_softmax_compute_time(input_size, num_cores):
    if num_cores == 1:
        table = softmax_tables[0]
    elif num_cores == 2:
        table = softmax_tables[1]
    elif num_cores == 4:
        table = softmax_tables[2]
    else:
        raise ValueError("Unsupported number of cores")
    compute_time = table[table['input_size'] == input_size]['compute_time(nsec)'].values[0] / 1e3
    return compute_time

def get_memory_access_time(time, tile_idx, type, addr):
    df = pd.DataFrame({
        'time': [time],
        'tile_idx': [tile_idx],
        'type': [type],
        'data': [addr.annotation],
        'base1': [addr.base[0]],
        'base2': [addr.base[1]],
        'offset1': [addr.offset[0]],
        'offset2': [addr.offset[1]],
    })
    df.to_csv('memory_request.csv')

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
    def __init__(self, start_time, end_time, tile_idx, type, addr):
        self.start_time = start_time
        self.end_time = end_time
        self.tile_idx = tile_idx
        self.type = type # e.g. 'load', 'store'
        self.addr = addr # TileAddress

class NoCTrace:
    def __init__(self, start_time, end_time, src_tile_idx, dest_tile_idx, type, addr):
        self.start_time = start_time
        self.end_time = end_time
        self.src_tile_idx = src_tile_idx
        self.dest_tile_idx = dest_tile_idx
        self.type = type # e.g. 'unicast'
        self.addr = addr # TileAddress

class ComputeTrace:
    def __init__(self, start_time, end_time, tile_idx, workload):
        self.start_time = start_time
        self.end_time = end_time
        self.tile_idx = tile_idx
        self.workload = workload

class SliceMatmul:
    def __init__(self, m, k, n, addrA, addrB, annotationRes, transpose=False):
        self.m = m
        self.k = k
        self.n = n
        self.addrA = addrA
        self.addrB = addrB
        self.transpose = transpose
        self.addrRes = TileAddress(
            (addrB.base[0] if not transpose else addrB.base[1], addrA.base[1]),
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
    def __init__(self, m, k, n, annotations, tile_size, transpose=False):
        self.m = m
        self.k = k
        self.n = n
        self.annotations = annotations # (e.g. {'A': 'X', 'B': 'Wq', 'Res': 'Q'})
        self.tile_size = tile_size
        self.m_pad = (m + tile_size[0] - 1) // tile_size[0] * tile_size[0]
        self.n_pad = (n + tile_size[1] - 1) // tile_size[1] * tile_size[1]
        self.m_split = self.m_pad // tile_size[0]
        self.n_split = self.n_pad // tile_size[1]
        self.transpose = transpose # whether to transpose the matB
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
                if self.transpose:
                    addrB = TileAddress(
                        (0, j * self.tile_size[1]), 
                        (self.k, self.tile_size[1]), 
                        self.annotations['B']
                    )
                slice = SliceMatmul(
                    m=self.tile_size[0],
                    k=self.k,
                    n=self.tile_size[1],
                    addrA=addrA,
                    addrB=addrB,
                    annotationRes=self.annotations['Res'],
                    transpose=self.transpose
                )
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

    def allocate(self, addr, access_time):
        self.get(addr.offset[0] * addr.offset[1])
        self.allocate_list.append((access_time, addr))

    def deallocate(self, addr):
        if addr in self.allocate_list:
            self.put(addr.offset[0] * addr.offset[1])
            for i, (_, allocated_addr) in enumerate(self.allocate_list):
                if allocated_addr == addr:
                    del self.allocate_list[i]
                    break

    def update_access_time(self, addr, access_time):
        for i, (_, allocated_addr) in enumerate(self.allocate_list):
            if allocated_addr == addr:
                self.allocate_list[i] = (access_time, allocated_addr)
                break

    def utilization(self):
        return self.level() / self.size

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
    def __init__(self, env, num_tiles, schedule_method, compute_graph):
        self.env = env
        self.tiles = [RISCVTile(env, i) for i in range(num_tiles)]
        self.global_memory = Memory(env, GLOBAL_MEMORY_SIZE, init=0)
        self.global_memory_dma = simpy.Resource(env, capacity=1)
        self.compute_graph = compute_graph
        self.schedule_method = schedule_method

    def search_parent(self, node):
        parents = list(self.compute_graph.predecessors(node))
        return parents
    
    def search_overlap(self, addr):
        overlap_list = []
        for core in self.tiles:
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

    def schedule(self, noc_trace, method='min_traffic'):
        if method == 'min_traffic':
            tile_idx = 0
            # Q, K, V projections
            if len(noc_trace) == 0:
                min_task_num = float('inf')
                for i, tile in enumerate(self.tiles):
                    if tile.task_num < min_task_num:
                        min_task_num = tile.task_num
                        tile_idx = i
            else:
                core_size_map = {}
                for trace in noc_trace:
                    if trace.tile_id is not None:
                        core_size_map[trace.tile_id] = core_size_map.get(trace.tile_id, 0) + trace.addrRes.offset[0] * trace.addrRes.offset[1]
                max_size = 0
                for i, core in enumerate(self.tiles):
                    if core.tile_id in core_size_map:
                        size = core_size_map[core.tile_id]
                    else:
                        size = 0
                    if size > max_size:
                        max_size = size
                        tile_idx = i
                    elif size == max_size:
                        if core.task_num < self.tiles[tile_idx].task_num:
                            tile_idx = i
        else:
            tile_idx = 0
            # Q, K, V projections
            min_task_num = float('inf')
            for i, core in enumerate(self.tiles):
                if core.task_num < min_task_num:
                    min_task_num = core.task_num
                    tile_idx = i

        return tile_idx
    
    def execute(self, workload, num_cores): # workload is a slice of a Matmul or Softmax
        workload.complete_event = self.env.event()
        memory_trace, noc_trace = self.analyze_workload(workload)
        for trace in noc_trace:
            yield trace.complete_event

        if self.schedule_method == 'min_traffic':
            tile_idx = self.schedule(noc_trace)
        else:
            tile_idx = self.schedule(noc_trace, method='min_idle')
        if isinstance(workload, SliceMatmul):
            self.tiles[tile_idx].task_num += 1
        
        with self.global_memory_dma.request() as global_request:
            yield global_request
            with self.tiles[tile_idx].dma_unit.request() as request:
                yield request
                for trace in memory_trace:
                    if trace[1] in self.tiles[tile_idx].local_memory.allocate_list:
                        continue
                    start_time = self.env.now
                    yield self.env.timeout(trace[1].offset[0]*trace[1].offset[1] / 1e3)
                    memory_record.append(MemoryTrace(start_time, self.env.now, tile_idx, 'load', trace[1]))
                    self.tiles[tile_idx].local_memory.allocate(trace[1], self.env.now)

        with self.tiles[tile_idx].noc_unit.request() as request:
            yield request
            for trace in noc_trace:
                if trace.tile_id == tile_idx:
                    continue
                if trace.addrRes in self.tiles[tile_idx].local_memory.allocate_list:
                    continue
                start_time = self.env.now
                yield self.env.timeout(trace.addrRes.offset[0]*trace.addrRes.offset[1] / 1e3)
                # yield self.env.timeout(trace.addrRes.offset[0]*trace.addrRes.offset[1] / 1e2)
                noc_record.append(NoCTrace(start_time, self.env.now, trace.tile_id, tile_idx, 'unicast', trace.addrRes))
                self.tiles[tile_idx].local_memory.allocate(trace.addrRes, self.env.now)

        with self.tiles[tile_idx].compute_unit.request() as request:
            workload.state = 'executing'
            yield request
            start_time = self.env.now
            if isinstance(workload, SliceMatmul):
                compute_time = get_gemm_compute_time(workload.addrA.offset[1], (workload.addrB.offset[0] if not workload.transpose else workload.addrB.offset[1]), workload.addrA.offset[0], num_cores)

            elif isinstance(workload, SliceSoftmax):
                compute_time = get_softmax_compute_time(workload.addrIn.offset[0], num_cores)
            yield self.env.timeout(compute_time)
            compute_record.append(ComputeTrace(start_time, self.env.now, tile_idx, workload))
            workload.state = 'completed'
            workload.tile_id = tile_idx
            self.tiles[tile_idx].local_memory.allocate(workload.addrRes, self.env.now)

        if isinstance(workload, SliceMatmul):
            self.tiles[tile_idx].task_num -= 1
        workload.complete_event.succeed()
        if workload.addrRes.annotation == 'Output':
            with self.global_memory_dma.request() as global_request:
                yield global_request
                with self.tiles[tile_idx].dma_unit.request() as request:
                    yield request
                    start_time = self.env.now
                    yield self.env.timeout(workload.addrRes.offset[0]*workload.addrRes.offset[1] / 1e3)
                    memory_record.append(MemoryTrace(start_time, self.env.now, tile_idx, 'store', workload.addrRes))

RANDOM_SEED = 42

class Attention:
    def __init__(self, seqlen, dim, config):
        self.seqlen = seqlen
        self.dim = dim

        # Project Q, K, VddrRes.offset
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
            tile_size=config.get('qk_matmul_tile_size', (16, 16)),
            transpose=True  # QK^T
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


def dump_event_trace(trace_list, filename, trace_type):
    with open(filename, 'w') as f:
        if trace_type == 'memory':
            f.write('start_time,end_time,tile_idx,type,base1,base2,offset1,offset2\n')
            for trace in trace_list:
                f.write(f"{trace.start_time},{trace.end_time},{trace.tile_idx},{trace.type},"
                        f"{trace.addr.base[0]},{trace.addr.base[1]},"
                        f"{trace.addr.offset[0]},{trace.addr.offset[1]}\n")
        elif trace_type == 'noc':
            f.write('start_time,end_time,src_tile_idx,dest_tile_idx,type,base1,base2,offset1,offset2\n')
            for trace in trace_list:
                f.write(f"{trace.start_time},{trace.end_time},{trace.src_tile_idx},{trace.dest_tile_idx},"
                        f"{trace.type},{trace.addr.base[0]},{trace.addr.base[1]},"
                        f"{trace.addr.offset[0]},{trace.addr.offset[1]}\n")
        else:
            f.write('start_time,end_time,tile_idx,type\n')
            for trace in trace_list:
                if isinstance(trace.workload, SliceMatmul):
                    f.write(f"{trace.start_time},{trace.end_time},{trace.tile_idx},matmul\n")
                elif isinstance(trace.workload, SliceSoftmax):
                    f.write(f"{trace.start_time},{trace.end_time},{trace.tile_idx},softmax\n")

def dump_trace_file():
    global memory_record, noc_record, compute_record
    with open('memory_trace.csv', 'w') as f:
        f.write('time,tile_idx,type,data,base1,base2,offset1,offset2\n')
        for trace in memory_record:
            f.write(f"{trace.start_time},{trace.tile_idx},{trace.type},{trace.addr.annotation},"
                    f"{trace.addr.base[0]},{trace.addr.base[1]},"
                    f"{trace.addr.offset[0]},{trace.addr.offset[1]}\n")

def evaluation(config):
    start = time.time()
    env = simpy.Environment()
    attn = Attention(seqlen=config['seqnum'], dim=config['dim'], config=config)
    processor = RISCVMultiprocessor(env, num_tiles=config['num_tiles'], schedule_method=config['schedule_method'], compute_graph=attn.graph)
    topological_sort = list(nx.topological_sort(attn.graph))
    for node in topological_sort:
        if node.addrRes.annotation == 'Q':
            num_cores = config['proj_q_core_num']
        elif node.addrRes.annotation == 'K':
            num_cores = config['proj_k_core_num']
        elif node.addrRes.annotation == 'V':
            num_cores = config['proj_v_core_num']
        elif node.addrRes.annotation == 'QK':
            num_cores = config['qk_matmul_core_num']
        elif node.addrRes.annotation == 'Softmax':
            num_cores = config['softmax_core_num']
        elif node.addrRes.annotation == 'Output':
            num_cores = config['qkv_matmul_core_num']
        env.process(processor.execute(node, num_cores))
    env.run()
    end = time.time()
    return env.now, end - start


def create_event_meta_data(num_tiles):
    events = []
    for tile_idx in range(num_tiles):
        process_dict = { "ph": "M", "pid": tile_idx, "tid": 0, "name": "process_name", "args": {"name": f'Tile {tile_idx}'}}
        compute_dict = { "ph": "M", "pid": tile_idx, "tid": 1, "name": "thread_name", "args": {"name": "Compute Unit"}}
        memory_dict = { "ph": "M", "pid": tile_idx, "tid": 2, "name": "thread_name", "args": {"name": "Memory Unit"}}
        noc_dict = { "ph": "M", "pid": tile_idx, "tid": 3, "name": "thread_name", "args": {"name": "NoC Unit"}}
        events.append(process_dict)
        events.append(compute_dict)
        events.append(memory_dict)
        events.append(noc_dict)
    return events

def google_trace_gen(tile_size, num_tiles, method):
    global memory_record, compute_record, noc_record

    events = create_event_meta_data(num_tiles)

    for tile_idx in range(num_tiles):
        core_compute_trace = [trace for trace in compute_record if trace.tile_idx == tile_idx]
        core_memory_trace = [trace for trace in memory_record if trace.tile_idx == tile_idx]
        core_noc_trace = [trace for trace in noc_record if trace.dest_tile_idx == tile_idx]
        
        for trace in core_compute_trace:
            event_dict = {"cat": f'{trace.workload.addrRes}', "ph": "X", "ts": round_decimal(trace.start_time), "pid": tile_idx, "tid": 1, "dur": round_decimal(trace.end_time - trace.start_time)}
            if trace.workload.addrRes.annotation == 'Output':
                event_dict["name"] = "Output"
            elif trace.workload.addrRes.annotation == 'QK':
                event_dict["name"] = "QK^T"
            elif trace.workload.addrRes.annotation == 'Softmax':
                event_dict["name"] = "Softmax"
            elif trace.workload.addrRes.annotation == 'Q':
                event_dict["name"] = "Q Projection"
            elif trace.workload.addrRes.annotation == 'K':
                event_dict["name"] = "K Projection"
            elif trace.workload.addrRes.annotation == 'V':
                event_dict["name"] = "V Projection"
            events.append(event_dict)
        for trace in core_memory_trace:
            event_dict = {"cat": f'{trace.addr}', "ph": "X", "ts": round_decimal(trace.start_time), "pid": tile_idx, "tid": 2, "dur": round_decimal(trace.end_time - trace.start_time)}
            if trace.type == 'load':
                event_dict["name"] = "Load"
            elif trace.type == 'store':
                event_dict["name"] = "Store"
            events.append(event_dict)
        for trace in core_noc_trace:
            event_dict = {"name":"unicast", "cat": f'{trace.src_tile_idx}->{trace.dest_tile_idx}_{trace.addr}', "ph": "X", "ts": round_decimal(trace.start_time), "pid": tile_idx, "tid": 3, "dur": round_decimal(trace.end_time - trace.start_time)}
            events.append(event_dict)
        json.dump(events, open(f'trace_{tile_size[0]}_{tile_size[1]}_{num_tiles}_{method}.json', 'w'), indent=4)

def draw_all(seqnum, dim):
    global memory_record, noc_record, compute_record
    X = np.arange(16, 65, 8)
    Y = np.arange(32, 169, 8)
    X, Y = np.meshgrid(X, Y)
    num_tiles = 8
    num_core = 1
    schedule_methods = ['min_idle', 'min_traffic']
    Z_idle = []
    Z_traffic = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for schedule_method in schedule_methods:
        for x, y in zip(X.flatten(), Y.flatten()):
            tile_size = (int(x), int(y))
            config = {
                'seqnum': seqnum,
                'dim': dim,
                'num_tiles': num_tiles,
                'proj_q_tile_size': tile_size,
                'proj_q_core_num': num_core,
                'proj_k_tile_size': tile_size,
                'proj_k_core_num': num_core,
                'proj_v_tile_size': tile_size,
                'proj_v_core_num': num_core,
                'qk_matmul_tile_size': tile_size,
                'qk_matmul_core_num': num_core,
                'qkv_matmul_tile_size': tile_size,
                'qkv_matmul_core_num': num_core,
                'softmax_core_num': num_core,
                'schedule_method': schedule_method  # or 'min_idle'
            }
            print(f"Running simulation with tile size ({x},{y}) and {num_tiles} tiles...")
            simulation_time, _ = evaluation(config)
            # Z.append(simulation_time)
            if schedule_method == 'min_idle':
                Z_idle.append(simulation_time)
            else:
                Z_traffic.append(simulation_time)

            # google_trace_gen(tile_size, num_tiles, schedule_method)
            memory_record = []
            compute_record = []
            noc_record = []

    Z_idle = np.array(Z_idle).reshape(X.shape)
    Z_traffic = np.array(Z_traffic).reshape(X.shape)
    vmin = min(Z_idle.min(), Z_traffic.min())
    vmax = max(Z_idle.max(), Z_traffic.max())
    im_idle = axes[0].imshow(Z_idle, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()],
                             aspect='auto', cmap='magma_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Simulation Time vs Tile Size (Min Idle)')
    axes[0].set_xlabel('Tile Size X')
    axes[0].set_ylabel('Tile Size Y')
    axes[0].set_xticks(X.flatten())
    axes[0].set_yticks(Y.flatten())
    axes[0].grid(False)
    fig.colorbar(im_idle, ax=axes[0], label='Simulation Time (microseconds)')

    im_traffic = axes[1].imshow(Z_traffic, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()],
                                aspect='auto', cmap='magma_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('Simulation Time vs Tile Size (Min Traffic)')
    axes[1].set_xlabel('Tile Size X')
    axes[1].set_ylabel('Tile Size Y')
    axes[1].set_xticks(X.flatten())
    axes[1].set_yticks(Y.flatten())
    axes[1].grid(False)
    fig.colorbar(im_traffic, ax=axes[1], label='Simulation Time (microseconds)')

    plt.tight_layout()
    plt.savefig(f'simulation_time_heatmap_{num_tiles}_{num_core}_{dim}.png')
    # np.savez('simulation_time_data.npz', X=X, Y=Y, Z_idle=Z_idle, Z_traffic=Z_traffic)

def main():
    global memory_record, noc_record, compute_record

    draw_all(seqnum=128, dim=128)
    draw_all(seqnum=256, dim=256)
    draw_all(seqnum=512, dim=512)

    # num_tiles = 8
    # num_core = 1
    # schedule_methods = ['min_idle', 'min_traffic']
    # # tile_size = (32, 64)
    # # tile_sizes = [(16, 64), (24, 64), (32, 64), (40, 64), (48, 64), (56, 64), (64, 64)]
    # tile_sizes = [(48, 160)]
    # plt.figure(figsize=(10, 6))
    # for schedule_method in schedule_methods:
    #     simulation_times = []
    #     for tile_size in tile_sizes:
    #         config = {
    #             'seqnum': SEQLEN,  # or 256, 512
    #             'dim': DIM,  # or 256, 512
    #             'num_tiles': num_tiles,
    #             'proj_q_tile_size': tile_size,
    #             'proj_q_core_num': num_core,
    #             'proj_k_tile_size': tile_size,
    #             'proj_k_core_num': num_core,
    #             'proj_v_tile_size': tile_size,
    #             'proj_v_core_num': num_core,
    #             'qk_matmul_tile_size': tile_size,
    #             'qk_matmul_core_num': num_core,
    #             'qkv_matmul_tile_size': tile_size,
    #             'qkv_matmul_core_num': num_core,
    #             'softmax_core_num': num_core,
    #             'schedule_method':  schedule_method  # or 'min_idle'
    #         }
    #         print(f"Running simulation with tile size {tile_size} and {num_tiles} tiles...")
    #         simulation_time, _ = evaluation(config)
    #         simulation_times.append((num_core, simulation_time))
    #         google_trace_gen(tile_size, num_tiles, schedule_method)
    #         dump_trace_file()
    #         memory_record = []
    #         compute_record = []
    #         noc_record = []

        # plt.plot([size[0] for size in simulation_times], [size[1] for size in simulation_times], marker='o', label=f'Schedule: {schedule_method}')
        # plt.plot([size[0][0] for size in simulation_times], [size[1] for size in simulation_times], marker='o', label=f'Schedule: {schedule_method}')
 
    # plt.title('Simulation Time vs Number of Cores')
    # plt.xlabel('Core num')
    # plt.ylabel('Simulation Time (seconds)')
    # plt.xticks([size[0] for size in simulation_times]) 
    # plt.legend()
    # plt.grid()
    # plt.savefig('simulation_time_vs_num_cores.png')


    # tile_size = [(32, 64)]
    # num_cores = [1, 2, 4, 8, 16, 32, 64, 128]
    # simulation_times = []
    # num_core = 1
    # for num_tile in num_cores:
    #     config = {
    #         'num_tiles': num_tile,
    #         'proj_q_tile_size': tile_size[0],
    #         'proj_q_core_num': num_core,
    #         'proj_k_tile_size': tile_size[0],
    #         'proj_k_core_num': num_core,
    #         'proj_v_tile_size': tile_size[0],
    #         'proj_v_core_num': num_core,
    #         'qk_matmul_tile_size': tile_size[0],
    #         'qk_matmul_core_num': num_core,
    #         'qkv_matmul_tile_size': tile_size[0],
    #         'qkv_matmul_core_num': num_core,
    #         'softmax_core_num': num_core,
    #         'schedule_method': 'min_traffic'
    #     }
    #     print(f"Running simulation with tile size {tile_size[0]} and {num_tile} tiles...")
    #     simulation_time, _ = evaluation(config)
    #     simulation_times.append((num_tile, simulation_time))

    # # draw histogram
    # plt.figure(figsize=(10, 6))
    # plt.bar([str(size[0]) for size in simulation_times], [size[1] for size in simulation_times])
    # plt.title('Simulation Time vs Number of Tiles')
    # plt.xlabel('Number of Tiles')
    # plt.ylabel('Simulation Time (seconds)')
    # plt.xticks(rotation=45)
    # plt.grid(axis='y')
    # # plt.show()
    # plt.savefig('simulation_time_vs_num_tiles.png')


if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from collections import namedtuple

# MatmulCompute Event: addr -> addrA, addrB, addrRes
# SoftmaxCompute Event: addr -> addrIn, addrOut
# NoCUnicast Event
# GlobalMemory Load Event
# GlobalMemory Store Event
# Event = namedtuple('Event', ['time', 'tile_id', 'event_type', 'data', 'addr'])

# EventType = GlobalMemory, NoC, Compute, HighLevel

# class Address:
#     def __int__ (self, base_addr, offset):
#         self.base_addr = base_addr #(i, j)
#         self.offset = offset # (width, height)

class Event:
    def __init__(self, time, event_type):
        self.time = time
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time

class ComputeEvent(Event):
    def __init__(self, time, op_name, tile_id, data, addr, duration):
        super().__init__(time, 'compute')
        self.op_name = op_name
        self.tile_id = tile_id
        self.duration = duration
        self.data = data # Matmul: (dataA, dataB, dataRes), Softmax: (dataIn, dataOut)
        self.addr = addr # Matmul: (addrA, addrB, addrRes), Softmax: (addrIn, addrOut)

class GlobalMemoryEvent(Event):
    def __init__(self, time, op_name, tile_id, data, addr, callback, duration):
        super().__init__(time, 'global_memory')
        self.op_name = op_name # 'load' or 'store'
        self.tile_id = tile_id
        self.data = data #list of data names
        self.addr = addr #list of addresses
        self.callback = callback
        self.duration = duration

class NoCEvent(Event):
    def __init__(self, time, op_name, tile_id, dest_tile_id, data, addr, duration):
        super().__init__(time, 'NoC')
        self.op_name = op_name
        self.tile_id = tile_id
        self.data = data
        self.addr = addr
        self.duration = duration
        self.dest_tile_id = dest_tile_id

class CallbackEvent(Event):
    def __init__(self, time, callback):
        super().__init__(time, 'callback')
        self.callback = callback

# BENCHMARK_GEMM = "/home/aster/Workspace/Lab447/RiscVSim/gemm_rvv_v5.csv"
# BENCHMARK_SOFTMAX = "/home/aster/Workspace/Lab447/RiscVSim/softmax_cycles_v1.csv"

NUM_TILES = 256

def transpose(addr):
    return [addr[1], addr[0], addr[3], addr[2]]

def size(addr):
    return (addr[2] - addr[0]) * (addr[3] - addr[1])

def overlap(addr1, addr2):
    addr_ans = [-1, -1, -1, -1]
    if addr1[0] < addr2[2] and addr1[2] > addr2[0] and addr1[1] < addr2[3] and addr1[3] > addr2[1]:
        addr_ans[0] = max(addr1[0], addr2[0])
        addr_ans[1] = max(addr1[1], addr2[1])
        addr_ans[2] = min(addr1[2], addr2[2])
        addr_ans[3] = min(addr1[3], addr2[3])
    return addr_ans if addr_ans[0] != -1 else None

def cover(addr, addr_set):
    lx, ly, rx, ry = addr
    area = (rx - lx) * (ry - ly)
    covered_area = 0
    for i in addr_set:
        tlx, tly, brx, bry = i
        inter_lx = max(lx, tlx)
        inter_ly = max(ly, tly)
        inter_rx = min(rx, brx)
        inter_ry = min(ry, bry)

        if inter_lx < inter_rx and inter_ly < inter_ry:
            covered_area += (inter_rx - inter_lx) * (inter_ry - inter_ly)
    return covered_area == area

class Tile:
    def __init__(self, tile_id):
        self.tile_id = tile_id
        self.compute_time = 0
        self.dma_time = 0
        self.local_memory = {}
        self.allocating = {}

    def add_compute_time(self, duration):
        self.compute_time += duration

    def schedule(self, event):
        resEvent = None
        if event.event_type == 'compute':
            actual_time = max(self.compute_time, event.time)
            self.compute_time += event.duration
            resEvent = ComputeEvent(actual_time, event.op_name, self.tile_id, event.data, event.addr, event.duration)
        elif event.event_type == 'global_memory':
            actual_time = max(self.dma_time, event.time)
            self.dma_time += event.duration
            resEvent = GlobalMemoryEvent(actual_time, event.op_name, self.tile_id, event.data, event.addr, event.callback, event.duration)
        else:
            raise ValueError(f"Unknown event type: {event.event_type}")
        return resEvent
    
    def is_available(self, time, event_type):
        if event_type == 'compute':
            return self.compute_time <= time
        elif event_type == 'global_memory':
            return self.dma_time <= time
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def _prelocate(self, data, addr):
        if data not in self.allocating:
            self.allocating[data] = []
        self.allocating[data].append(addr)

    def _remove(self, data, addr):
        if data in self.allocating and addr in self.allocating[data]:
            self.allocating[data].remove(addr)
            if not self.allocating[data]:
                del self.allocating[data]

    def _allocate(self, data, addr):
        if data not in self.local_memory:
            self.local_memory[data] = []
        for existing_addr in self.local_memory[data]:
            if overlap(existing_addr, addr):
                raise ValueError(f"Address {addr} overlaps with existing address {existing_addr} in tile {self.tile_id}")
        self.local_memory[data].append(addr)

    def _get_memory_usage(self):
        usage = 0
        for op_name in self.local_memory:
            for addr in self.local_memory[op_name]:
                usage += size(addr)
        return usage
    
class EventSimulator:
    def __init__(self, num_tiles):
        self.num_tiles = num_tiles
        self.available_tiles = num_tiles
        self.tiles = [Tile(i) for i in range(num_tiles)]
        self.event_queue = []
        self.waiting_events = []
    
        
    def allocate(self, tile_id, data, addr):
        if 0 <= tile_id and tile_id < len(self.tiles):
            self.tiles[tile_id]._allocate(data, addr)
        else:
            raise ValueError(f"Invalid tile_id: {tile_id}")

    def trace(self, query_data, query_addr):
        results = []
        for tile in self.tiles:
            if query_data in tile.local_memory:
                for addr in tile.local_memory[query_data]:
                    if overlap(addr, query_addr):
                        results.append((tile.tile_id, addr))
        return results

    def get_memory_usage(self):
        usage = {}
        for tile in self.tiles:
            usage[tile.tile_id] = tile._get_memory_usage()
        return usage

    def __str__(self):
        return "\n".join(f"Tile {tile.tile_id}: {tile.local_memory}" for tile in self.tiles)

    def add_event(self, event):
        sched_event = self.tiles[event.tile_id].schedule(event)
        heapq.heappush(self.event_queue, sched_event)

    def add_callback_event(self, event):
        heapq.heappush(self.event_queue, event)

    def run(self, config, seqlen, dim):
        matmul_q = MatMul(seqlen, dim, dim, (config['tile_size_m'], config['tile_size_n']), ('X', 'Wq', 'Q'))
        # matmul_k = MatMul(seqlen, dim, dim, (config['tile_size_m'], config['tile_size_n']), ('X', 'Wk', 'K'))
        # matmul_qk = MatMul(seqlen, dim, seqlen, (config['tile_size_m'], config['tile_size_n']), ('Q', 'K', 'QK^T'))
        # softmax = Softmax(seqlen, ('QK^T', 'Softmax'))
        # matmul_v = MatMul(seqlen, dim, dim, (config['tile_size_m'], config['tile_size_n']), ('X', 'Wv', 'V'))
        # matmul_qkv = MatMul(seqlen, seqlen, dim, (config['tile_size_m'], config['tile_size_n']), ('Softmax', 'V', 'QKV'))
    
        for idx, slice in enumerate(matmul_q.slices):
            addrA = slice.addrA
            dataA = slice.dataA
            addrB = slice.addrB
            dataB = slice.dataB
            tile_id = idx % config['proj_q_tiles']
            self.add_event(ComputeEvent(0, 'matmul', tile_id, (dataA, dataB, slice.dataRes), (addrA, addrB, slice.addrRes), 0))
            # print(f"Adding ComputeEvent for tile {tile_id}: {slice}")
            # print(f"Data A: {dataA}, Addr A: {addrA}, Data B: {dataB}, Addr B: {addrB}, Result Data: {slice.dataRes}, Result Addr: {slice.addrRes}")
            # self.add_event(GlobalMemoryEvent(0, 'load', tile_id, dataA, addrA, durationA))
            # self.add_event(GlobalMemoryEvent(0, 'load', tile_id, dataB, addrB, durationB))

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            if event.event_type == 'compute' and event.duration == 0:
                tile = self.tiles[event.tile_id]
                ready = True
                data_req = []
                addr_req = []
                if not cover(event.addr[0], tile.local_memory.get(event.data[0], [])) and not cover(event.addr[0], tile.allocating.get(event.data[0], [])):
                    ready = False
                    data_req.append(event.data[0])
                    addr_req.append(event.addr[0])
                    tile._prelocate(event.data[0], event.addr[0])
                if not cover(event.addr[1], tile.local_memory.get(event.data[1], [])) and not cover(event.addr[1], tile.allocating.get(event.data[1], [])):
                    ready = False
                    data_req.append(event.data[1])
                    addr_req.append(event.addr[1])
                    tile._prelocate(event.data[1], event.addr[1])
                if ready:
                    self.add_event(ComputeEvent(event.time, event.op_name, event.tile_id, event.data, event.addr, 100))
                else:
                    if len(data_req) == 2:
                        dur = 100
                    else:
                        dur = 50
                    self.add_event(GlobalMemoryEvent(event.time, 'load', event.tile_id, data_req, addr_req, event, dur))
                    self.waiting_events.append(event)

            elif event.event_type == 'compute':
                tile = self.tiles[event.tile_id]
                tile._allocate(event.data[2], event.addr[2])
                print(f"Executing Compute event: {event.op_name} on tile {event.tile_id} at time {event.time}")
                    
            elif event.event_type == 'global_memory':
                print(f"Executing GlobalMemoryEvent: {event.op_name} on tile {event.tile_id} at time {event.time}")
                tile = self.tiles[event.tile_id]
                self.add_callback_event(CallbackEvent(event.time + event.duration, event.callback))
                for data, addr in zip(event.data, event.addr):
                    if event.op_name == 'load':
                        tile._allocate(data, addr)
                        tile._remove(data, addr)

            elif event.event_type == 'callback':
                if event.callback.event_type == 'compute':
                    callback_event = event.callback
                    callback_event.time = event.time
                    self.add_event(callback_event)

class SliceMatMul:
    # addr: [leftup_x, leftup_y, rightdown_x, rightdown_y]
    def __init__(self, m, k, n, addrA, addrB, data_name):
        self.m = m
        self.k = k
        self.n = n
        self.addrA = addrA
        self.addrB = addrB
        self.addrRes = [addrB[0], addrA[1], addrB[2], addrA[3]]
        self.dataA = data_name[0]  # Name of matrix A
        self.dataB = data_name[1]  # Name of matrix B
        self.dataRes = data_name[2]  # Name of result matrix

    def __str__(self):
        return f"SliceMatMul(m={self.m}, k={self.k}, n={self.n}, addrA={self.addrA}, addrB={self.addrB}, addrRes={self.addrRes})"

class MatMul:
    def __init__(self, m, k, n, tile_size, data_name):
        self.m = m
        self.k = k
        self.n = n
        self.tile_size = tile_size # (m, n)
        self.m_pad = (m + tile_size[0] - 1) // tile_size[0] * tile_size[0]
        self.n_pad = (n + tile_size[1] - 1) // tile_size[1] * tile_size[1]
        self.m_split = self.m_pad // tile_size[0]
        self.n_split = self.n_pad // tile_size[1]
        self.addrA = [0, 0, self.m_pad, k]
        self.addrB = [0, 0, k, self.n_pad]
        self.data_name = data_name  # (name A, name B, name Res)
        self.slices = self.get_slices()

    def get_slices(self):
        slices = []
        for i in range(self.m_split):
            for j in range(self.n_split):
                addrA = [0, i * self.tile_size[0], self.m, (i + 1) * self.tile_size[0]]
                addrB = [j * self.tile_size[1], 0, (j + 1) * self.tile_size[1], self.k]
                slices.append(SliceMatMul(self.tile_size[0], self.k, self.tile_size[1], addrA, addrB, self.data_name))
        return slices
    
class SliceSoftmax:
    def __init__(self, seqlen, addrIn, addrOut, data_name):
        self.seqlen = seqlen
        self.addrIn = addrIn
        self.addrOut = addrOut
        self.dataIn = data_name[0]  # Name of input data
        self.dataOut = data_name[1]  # Name of output data

    def __str__(self):
        return f"SliceSoftmax(input_size={self.input_size}, seqlen={self.seqlen}, addrIn={self.addrIn}, addrOut={self.addrOut})"

class Softmax:
    def __init__(self, seqlen, data_name):
        self.seqlen = seqlen
        self.addrIn = [0, 0, seqlen, seqlen]
        self.addrOut = [0, 0, seqlen, seqlen]
        self.slices = self.get_slice()
        self.data_name = data_name

    def get_slice(self):
        slices = []
        for i in range(self.seqlen):
            addrIn  = [i, 0, i, self.seqlen]
            addrOut = [i, 0, i, self.seqlen]
            slices.append(SliceSoftmax(self.input_size, self.seqlen, addrIn, addrOut, self.data_name))
        return slices

class Attention:
    def __init__(self, seqlen, dim):
        self.seqlen = seqlen
        self.dim = dim

def main():
    seqlen = 320
    dim = 256
    config = {
        'proj_q_tiles': 16,  # Number of tiles for Q projection
        'proj_q_tile_size_m': 64,
        'proj_q_tile_size_n': 64,
        'proj_k_tiles': 16,  # Number of tiles for K projection
        'proj_k_tile_size_m': 64,
        'proj_k_tile_size_n': 64,
        'proj_v_tiles': 16,  # Number of tiles for V projection
        'proj_v_tile_size_m': 64,
        'proj_v_tile_size_n': 64,
        'qk_tiles': 16,
        'qk_tile_size_m': 64,
        'qk_tile_size_n': 64,
        'qkv_tiles': 16,
        'qkv_tile_size_m': 64,
        'qkv_tile_size_n': 64,
        'softmax_tiles': 16,
    }
    n_tiles = 32  # Number of tiles to use
    simulator = EventSimulator(NUM_TILES)
    simulator.run(config, seqlen, dim)

if __name__ == "__main__":
    main()
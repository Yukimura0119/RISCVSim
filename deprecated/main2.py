import pandas as pd
import matplotlib.pyplot as plt

# BENCHMARK_GEMM = "/home/aster/Workspace/Lab447/RiscVSim/gemm_rvv_v5.csv"
# BENCHMARK_SOFTMAX = "/home/aster/Workspace/Lab447/RiscVSim/softmax_cycles_v1.csv"

NUM_TILES = 256

def get_tile_set(num_tiles):
    tile_set = {}
    for i in range(num_tiles):
        tile_set[i] = 0
    return tile_set

def overlap(addr1, addr2):
    addr_ans = [-1, -1, -1, -1]
    if addr1[0] < addr2[2] and addr1[2] > addr2[0] and addr1[1] < addr2[3] and addr1[3] > addr2[1]:
        addr_ans[0] = max(addr1[0], addr2[0])
        addr_ans[1] = max(addr1[1], addr2[1])
        addr_ans[2] = min(addr1[2], addr2[2])
        addr_ans[3] = min(addr1[3], addr2[3])
    return addr_ans if addr_ans[0] != -1 else None

class Tile:
    def __init__(self, tileID):
        self.tileID = tileID
        self.local_memory = {}

    def _allocate(self, addr, op_name):
        if op_name not in self.local_memory:
            self.local_memory[op_name] = []
        for existing_addr in self.local_memory[op_name]:
            if overlap(existing_addr, addr):
                raise ValueError(f"Address {addr} overlaps with existing address {existing_addr} in tile {self.tileID} for operation {op_name}")
        self.local_memory[op_name].append(addr)

    def _get_memory_usage(self):
        usage = 0
        for op_name in self.local_memory:
            for addr in self.local_memory[op_name]:
                usage += size(addr)
        return usage

class TileManager:
    def __init__(self, num_tiles):
        self.num_tiles = num_tiles
        self.tiles = [Tile(i) for i in range(num_tiles)]

    def allocate(self, tileID, addr, op_name):
        if 0 <= tileID and tileID < len(self.tiles):
            self.tiles[tileID]._allocate(addr, op_name)
        else:
            raise ValueError(f"Invalid tileID: {tileID}")

    def trace(self, query_addr, query_op_name):
        results = []
        for tile in self.tiles:
            if query_op_name in tile.local_memory:
                for addr in tile.local_memory[query_op_name]:
                    if overlap(addr, query_addr):
                        results.append((tile.tileID, addr))
        return results

    def get_memory_usage(self):
        usage = {}
        for tile in self.tiles:
            usage[tile.tileID] = tile._get_memory_usage()
        return usage

    def __str__(self):
        return "\n".join(f"Tile {tile.tileID}: {tile.local_memory}" for tile in self.tiles)

class SliceMatMul:
    # addr: [leftup_x, leftup_y, rightdown_x, rightdown_y]
    def __init__(self, m, k, n, addrA, addrB, tileID):
        self.m = m
        self.k = k
        self.n = n
        self.tileID = tileID
        self.addrA = addrA
        self.addrB = addrB
        self.addrC = [addrB[0], addrA[1], addrB[2], addrA[3]]

    def __str__(self):
        return f"SliceMatMul(m={self.m}, k={self.k}, n={self.n}, tileID={self.tileID}, addrA={self.addrA}, addrB={self.addrB}, addrC={self.addrC})"
class MatMul:
    def __init__(self, m, k, n, num_split):
        self.m = m
        self.k = k
        self.n = n
        self.num_split = num_split
        self.num_split_m = int(self.num_split**0.5)
        self.num_split_n = int(self.num_split**0.5)
        self.m_pad = (m + self.num_split_m - 1) // self.num_split_m * self.num_split_m
        self.n_pad = (n + self.num_split_n - 1) // self.num_split_n * self.num_split_n
        self.m_split = self.m_pad // self.num_split_m
        self.n_split = self.n_pad // self.num_split_n
        self.addrA = [0, 0, self.m_pad, self.k]
        self.addrB = [0, 0, self.k, self.n_pad]
        self.slices = self.get_slices()

    def get_slices(self):
        slices = []
        for i in range(self.num_split_m):
            for j in range(self.num_split_n):
                addrA = [0, i * self.m_split, self.k, (i + 1) * self.m_split]
                addrB = [j * self.n_split, 0, (j + 1) * self.n_split, self.k]
                slices.append(SliceMatMul(self.m_split, self.k, self.n_split, addrA, addrB, None))
        return slices
    
    # def set_s
    
    def schedule(self, tileManager: TileManager, matA_op_name, matB_op_name, matC_op_name, is_transpose=False):
        traffic_list = [] # From, to, addr, op_name
        tile_set = get_tile_set(tileManager.num_tiles)
        for slice in self.slices:
            # Reset tile_set for each slice
            for i in tile_set:
                tile_set[i] = 0
            trace_A = tileManager.trace(slice.addrA, matA_op_name)
            # print(f"{matA_op_name} trace_A({slice.addrA}):")
            for tileID, addr in trace_A:
                # print(f"Tile {tileID}: {addr}")
                if tileID in tile_set:
                    tile_set[tileID] += size(addr)
            if is_transpose:
                trace_B = tileManager.trace(transpose(slice.addrB), matB_op_name)
            else:
                trace_B = tileManager.trace(slice.addrB, matB_op_name)
            for tileID, addr in trace_B:
                if tileID in tile_set:
                    tile_set[tileID] += size(addr)

            max_tileID = max(tile_set, key=tile_set.get)

            # Get Traffic for addrA, addrB
            for tileID, addr in trace_A:
                if tileID != max_tileID:
                    traffic_list.append((tileID, max_tileID, addr, matA_op_name))
                elif tileID == max_tileID and size(addr) == size(slice.addrA):
                    traffic_list = []
                    break
            traffic_list_backup = traffic_list.copy()
            for tileID, addr in trace_B:
                if tileID != max_tileID:
                    traffic_list.append((tileID, max_tileID, addr, matB_op_name))
                elif tileID == max_tileID and size(addr) == size(slice.addrA):
                    traffic_list = traffic_list_backup
                    break
            slice.tileID = max_tileID
            tile_set.pop(max_tileID)
            tileManager.allocate(max_tileID, slice.addrC, matC_op_name)
        for _, to_tile, addr, op_name in traffic_list:
            tileManager.allocate(to_tile, addr, op_name)
        for traffic in traffic_list:
            print(f"Traffic: From tile {traffic[0]} to tile {traffic[1]} for operation {traffic[3]} at address {traffic[2]}")
        return traffic_list

class SliceSoftmax:
    def __init__(self, input_size, seqlen, addrIn, addrOut, tileID=None):
        self.input_size = input_size
        self.seqlen = seqlen
        self.addrIn = addrIn
        self.addrOut = addrOut
        self.tileID = tileID

    def __str__(self):
        return f"SliceSoftmax(input_size={self.input_size}, seqlen={self.seqlen}, addrIn={self.addrIn}, addrOut={self.addrOut})"

class Softmax:
    def __init__(self, input_size, seqlen, num_split):
        self.input_size = input_size
        self.seqlen = seqlen
        self.addrIn = [0, 0, input_size, seqlen]
        self.addrOut = [0, 0, input_size, seqlen]
        self.num_split = num_split
        self.slices = self.get_slice()

    def schedule(self, tileManager: TileManager, input_op_name, output_op_name):
        traffic_list = []
        tile_set = get_tile_set(tileManager.num_tiles)
        for slice in self.slices:
            for i in tile_set:
                tile_set[i] = 0
            trace_in = tileManager.trace(slice.addrIn, input_op_name)
            for tileID, addr in trace_in:
                if tileID in tile_set:
                    tile_set[tileID] += size(addr)
            max_tileID = max(tile_set, key=tile_set.get)
            # Traffic_QK
            for tileID, addr in trace_in:
                if tileID != max_tileID:
                    traffic_list.append((tileID, max_tileID, addr, input_op_name))
            slice.tileID = max_tileID
            tile_set.pop(max_tileID)
            tileManager.allocate(max_tileID, slice.addrOut, output_op_name)
        for _, to_tile, addr, op_name in traffic_list:
            tileManager.allocate(to_tile, addr, op_name)
        return traffic_list
    
    def get_slice(self):
        slice_size = (self.seqlen + self.num_split - 1) // self.num_split
        slices = []
        for i in range(self.num_split):
            start = i * slice_size
            end = min((i + 1) * slice_size, self.seqlen)
            addrIn = [0, start, self.input_size, end]
            addrOut = [0, start, self.input_size, end]
            slices.append(SliceSoftmax(self.input_size, self.seqlen, addrIn, addrOut, i))
        return slices

def transpose(addr):
    return [addr[1], addr[0], addr[3], addr[2]]

def size(addr):
    return (addr[2] - addr[0]) * (addr[3] - addr[1])

class Attention:
    def __init__(self, seqlen, dim, num_tiles):
        self.seqlen = seqlen
        self.dim = dim
        self.num_tiles = num_tiles

    def get_cycles(self, tileManager: TileManager):
        compute_cycles = 0
        data_cycles = 0

        matmul_Q = MatMul(self.seqlen, self.dim, self.dim, self.num_tiles)
        for idx, slice in enumerate(matmul_Q.slices):
            tileManager.allocate(idx, slice.addrA, "X")
            tileManager.allocate(idx, slice.addrB, "Wq")
        # Read input data for matmul_Q
        data_cycles += size(matmul_Q.slices[0].addrA) + size(matmul_Q.slices[0].addrB)
        matmul_Q.schedule(tileManager, "X", "Wq", "Q")

        # simulate barrier
        if compute_cycles < data_cycles:
            compute_cycles = data_cycles

        compute_cycles += size(matmul_Q.slices[0].addrA) * size(matmul_Q.slices[0].addrB) // 200

        matmul_K = MatMul(self.seqlen, self.dim, self.dim, self.num_tiles)
        for idx, slice in enumerate(matmul_K.slices):
            tileManager.allocate(idx, slice.addrB, "Wk")
        data_cycles += size(matmul_K.slices[0].addrA) + size(matmul_K.slices[0].addrB)

        # simulate barrier
        if compute_cycles < data_cycles:
            compute_cycles = data_cycles
        matmul_K.schedule(tileManager, "X", "Wk", "K")
        compute_cycles += size(matmul_K.slices[0].addrA) * size(matmul_K.slices[0].addrB) // 200

        matmul_V = MatMul(self.seqlen, self.dim, self.dim, self.num_tiles)
        for idx, slice in enumerate(matmul_V.slices):
            tileManager.allocate(idx, slice.addrB, "Wv")

        # load wv when compute K
        data_cycles += size(matmul_V.slices[0].addrA) + size(matmul_V.slices[0].addrB)

        matmul_V.schedule(tileManager, "X", "Wv", "V")

        total_traffic = 0
        
        matmul_QK = MatMul(self.seqlen, self.dim, self.seqlen, self.num_tiles)
        traffic_list = matmul_QK.schedule(tileManager, "Q", "K", "QK", is_transpose=True)
        total_traffic += len(traffic_list)
        # compute K done, but load Wv is not done yet
        if compute_cycles < data_cycles:
            # wait for loading Wv done
            compute_cycles = data_cycles
        # start traffic
        data_cycles += len(traffic_list) * size(matmul_QK.slices[0].addrA)
        # compute V when moving data
        compute_cycles += size(matmul_QK.slices[0].addrA) * size(matmul_QK.slices[0].addrB) // 200

        # wait for traffic done
        if compute_cycles < data_cycles:
            compute_cycles = data_cycles
        # compute QK^T
        compute_cycles += size(matmul_QK.slices[0].addrA) * size(matmul_QK.slices[0].addrB) // 200
        # for traffic in traffic_list:
        #     print(f"QK^T: Traffic from tile {traffic[0]} to tile {traffic[1]} for operation {traffic[3]} at address {traffic[2]}")
        
        softmax = Softmax(self.dim, self.seqlen, self.num_tiles)
        traffic_list = softmax.schedule(tileManager, "QK", "softmax")
        total_traffic += len(traffic_list)

        # start traffic for softmax
        data_cycles += len(traffic_list) * size(softmax.slices[0].addrIn)
        # wait for traffic done
        if compute_cycles < data_cycles:
            compute_cycles = data_cycles
        # compute softmax
        compute_cycles += softmax.slices[0].input_size * self.seqlen // NUM_TILES // 400

        matmul_QKV = MatMul(self.seqlen, self.seqlen, self.dim, self.num_tiles)
        matmul_QKV.schedule(tileManager, "softmax", "V", "QKV")
        total_traffic += len(traffic_list)

        # start traffic for matmul_QKV
        data_cycles += len(traffic_list) * size(matmul_QKV.slices[0].addrA)
        # wait for traffic done
        if compute_cycles < data_cycles:
            compute_cycles = data_cycles
        # compute softmax
        compute_cycles += softmax.slices[0].input_size * self.seqlen // NUM_TILES // 400


        print(f"Total traffic: {total_traffic}")
        tileManager.get_memory_usage()
        for tileID, usage in tileManager.get_memory_usage().items():
            print(f"Tile {tileID} memory usage: {usage*2} bytes")
        return compute_cycles

def main():
    NUM_TILES = 8
    # num_split = 
    num_split = [16]
    res = []
    seqlen = 1024
    dim = 1024
    
    for x in num_split:
        tileManager = TileManager(x)
        attention = Attention(seqlen, dim, x)
        compute_cycles = attention.get_cycles(tileManager)
        res.append(compute_cycles)
        # print(f"Total compute cycles for attention with seqlen {seqlen} and dim {dim}: {compute_cycles}")

    # plt.plot(num_split, res)
    # plt.savefig("attention_cycles.png")
if __name__ == "__main__":
    main()
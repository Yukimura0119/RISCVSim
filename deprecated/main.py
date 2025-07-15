import pandas as pd

BENCHMARK_GEMM = "/home/aster/Workspace/Lab447/RiscVSim/gemm_rvv_v5.csv"
BENCHMARK_SOFTMAX = "/home/aster/Workspace/Lab447/RiscVSim/softmax_cycles_v1.csv"

class MatMul:
    def __init__(self, m, k, n, tile_size):
        self.tile_size = tile_size
        self.m = m
        self.k = k
        self.n = n
        self.m_pad = (m + tile_size - 1) // tile_size * tile_size
        self.k_pad = (k + tile_size - 1) // tile_size * tile_size
        self.n_pad = (n + tile_size - 1) // tile_size * tile_size
        self.cycle_per_tile = self.get_cycles()

    def get_cycles(self):
        df = pd.read_csv(BENCHMARK_GEMM)
        return df[df['input_size'] >= self.tile_size].iloc[0]['cycles']
        
    def get_num_tiles(self):
        num_tiles_m = self.m_pad // self.tile_size
        num_tiles_k = self.k_pad // self.tile_size
        num_tiles_n = self.n_pad // self.tile_size
        return num_tiles_m, num_tiles_k, num_tiles_n
    
    def get_total_cycles(self):
        num_tiles_m, num_tiles_n, num_tiles_k = self.get_num_tiles()
        total_cycles = num_tiles_m * num_tiles_k * num_tiles_n * self.cycle_per_tile
        return total_cycles.item()

class Softmax:
    def __init__(self, input_size):
        self.input_size = input_size
        self.cycle = self.get_cycles()
    
    def get_cycles(self):
        df = pd.read_csv(BENCHMARK_SOFTMAX)
        return df[df['input_size'] >= self.input_size].iloc[0]['fp32_cycles']


class Attention:
    def __init__(self, seqlen, dim, tile_size_dict):
        self.seqlen = seqlen
        self.tile_size_dict = tile_size_dict
        self.dim = dim
        self.cycle = self.get_cycles()

    def get_cycles(self):
        matmul_Q = MatMul(self.seqlen, self.dim, self.dim, self.tile_size_dict['Q'])
        matmul_K = MatMul(self.seqlen, self.dim, self.dim, self.tile_size_dict['K'])
        matmul_V = MatMul(self.seqlen, self.dim, self.dim, self.tile_size_dict['V'])
        matmul_QK = MatMul(self.seqlen, self.dim, self.seqlen, self.tile_size_dict['QK'])
        softmax = Softmax(self.dim)
        matmul_QKV = MatMul(self.seqlen, self.seqlen, self.dim, self.tile_size_dict['QKV'])
        total_cycles = (matmul_Q.get_total_cycles() +
                        matmul_K.get_total_cycles() +
                        matmul_V.get_total_cycles() +
                        matmul_QK.get_total_cycles() +
                        softmax.cycle * self.seqlen +
                        matmul_QKV.get_total_cycles())
        return total_cycles


def main():
    tile_size_dict = {
        'Q': 64,
        'K': 64,
        'V': 64,
        'QK' : 64,
        'QKV': 64,
    }
    seqlen = 12000
    dim = 130000
    attention = Attention(seqlen, dim, tile_size_dict)
    total_cycles = attention.cycle
    print(f"Total cycles for attention with seqlen {seqlen} and dim {dim}: {total_cycles}")

if __name__ == "__main__":
    main()
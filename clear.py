import pandas as pd

data_path = 'gemm_4cores_0729.csv'

df = pd.read_csv(data_path)

# M = 32, 64, 96, 128, 160, 192, 224, 256
# N = 32, 64, 96, 128, 160, 192, 224, 256
# K = 128, 256, 512, 1024, 2048

df = df[(df['M'] >= 32) & (df['M'] <= 256) &
		(df['N'] >= 32) & (df['N'] <= 256) &
		(df['K'] >= 128) & (df['K'] <= 2048)]

df = df[(df['M'] % 32 == 0) & (df['N'] % 32 == 0) & (df['K'] % 128 == 0)]

df.to_csv('gemm_4cores_0729_filtered.csv', index=False)
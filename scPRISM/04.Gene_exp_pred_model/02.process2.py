import torch
import scanpy as sc
#h5ad太大了，拆分成npy
data=sc.read_h5ad('./data/adata_process.h5ad')
data_x=data.X
import numpy as np
from tqdm import tqdm
for i in tqdm(range(data_x.shape[0])):
    row = np.array(data_x[i, :].todense(),dtype=np.float32)[0]
    np.save('./adata_process/'+str(i)+'.npy',row)

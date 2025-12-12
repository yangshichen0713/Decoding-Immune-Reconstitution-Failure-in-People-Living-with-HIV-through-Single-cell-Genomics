import scanpy as sc
import anndata as ad
import numpy as np
import scipy.sparse as sp
import gc 

# 读取原始数据
adata = sc.read_h5ad('/home/share/huadjyin/home/baiyong01/repo/hiv/pbmc_celltype_gzip.h5ad', backed='r')

# 创建新的 AnnData 对象并保存
raw_adata = ad.AnnData(
    X=adata.layers['counts'], 
    obs=adata.obs.copy(),
    var=adata.var.copy() 
)

sc.pp.normalize_total(raw_adata, target_sum=1e4)
sc.pp.log1p(raw_adata)

# 保存 normalize_log1p 数据
normalized_output_path = '/home/share/huadjyin/home/lutianyu/01HIV/02data/scrna/normalize_log1p.h5ad'
raw_adata.write(normalized_output_path, compression='gzip', compression_opts=6)
print(raw_adata)
print(f"Saved normalize_log1p")

# 重新读取原始数据以获取 HVG 信息
adata = sc.read_h5ad('/home/share/huadjyin/home/baiyong01/repo/hiv/pbmc_celltype_gzip.h5ad', backed='r')

# 提取 HVG 基因的表达矩阵
hvg_adata = raw_adata[:, adata.var['highly_variable_intersection']].copy()

# 保存 HVG 数据到 hvg.h5ad
hvg_output_path = '/home/share/huadjyin/home/lutianyu/01HIV/02data/scrna/hvg.h5ad'
hvg_adata.write(hvg_output_path, compression='gzip', compression_opts=6)
print(hvg_adata)
print(f"Saved HVG data")

import torch
import os
import scanpy as sc
import scanpy.external as sce
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.sparse import hstack, vstack, csr_matrix
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Dict, List
import warnings
import time
import shutil
import psutil
import gc
warnings.filterwarnings("ignore")


def init_data(rna_path: str, atac_path: str) -> Tuple[sc.AnnData, sc.AnnData, sc.AnnData]:
    """加载数据"""
    # 读取RNA数据
    rna = sc.read_h5ad(rna_path)
    # 读取ATAC数据
    atac = sc.read_h5ad(atac_path)
    

    return rna, atac

# def init_data(rna_path: str, atac_path: str, atac_path: str) -> Tuple[sc.AnnData, sc.AnnData, sc.AnnData]:
#     """加载数据"""
#     rna = sc.read_h5ad(rna_path)
#     atac = sc.read_h5ad(atac_path)
#     atac = sc.read_h5ad(atac_path)
    

#     return rna, atac, atac

def adata_drop_duplicates(adata):
    # 将adata.obs_names重命名为'matched_cell_i'，其中i为adata.n_obs的值
    adata.obs_names = [f'matched_cell_{i}' for i in range(adata.n_obs)]
    # 删除adata.obs中的重复行
    obs = adata.obs.drop_duplicates()
    # 根据obs.index筛选adata
    adata = adata[obs.index]
    # 将adata.obs设置为obs
    adata.obs = obs
    # 将adata.obs_names重命名为'matched_cell_i'，其中i为adata.n_obs的值
    adata.obs_names = [f'matched_cell_{i}' for i in range(adata.n_obs)]

    return adata

def process_celltype_sample(rna: sc.AnnData, atac: sc.AnnData, sample: str,
                           output_dir: str, n_neighbors: int = 50) -> Tuple[int, int]:
    """处理单个细胞类型和样本的匹配流程"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 筛选当前细胞类型和样本的数据
    rna_ct = rna[rna.obs['sample'] == sample]
    atac_ct = atac[atac.obs['people'] == sample]
    
    # 跳过细胞数不足的情况
    if len(rna_ct) < n_neighbors or len(atac_ct) < n_neighbors:
        print(f"Skipping {ct} {sample} due to insufficient cells", len(rna_ct), len(atac_ct))
        return 0, 0
    
    # KNN匹配
    try:
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(atac_ct.obsm['X_pca'])
        distances_rna, indices_rna = knn.kneighbors(rna_ct.obsm['X_pca'])
        
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(rna_ct.obsm['X_pca'])
        distances_atac, indices_atac = knn.kneighbors(atac_ct.obsm['X_pca'])
    except Exception as e:
        print(f"Error in KNN: {e}")
        return 0, 0
    

    # 构建代价矩阵（以 RNA 和 ATAC 匹配为例）
    cost_matrix = np.full((len(rna_ct), len(atac_ct)), np.inf)  # 初始化为无穷大

    for i, rna_cell in enumerate(rna_ct.obs_names):
        for j, dist in zip(indices_rna[i], distances_rna[i]):
            if dist < np.inf:  # 如果距离有效
                cost_matrix[i, j] = dist  # 填充 RNA 到 ATAC 的距离

    # 第二轮填充：ATAC 到 RNA（仅填充没有值的位置，或替换为较小的代价）
    for i, atac_cell in enumerate(atac_ct.obs_names):
        for j, dist in zip(indices_atac[i], distances_atac[i]):
            if dist < np.inf:  # 如果距离有效
                current_cost = cost_matrix[j, i]
                if current_cost < np.inf:  # 检查当前位置是否有值,有值代表MNN，距离相同
                    new_cost = current_cost / 2
                else:
                    new_cost = dist
                cost_matrix[j, i] = new_cost
    # 检查并处理无穷大值：用一个很大的值来代替无穷大
    cost_backup = cost_matrix.copy()
    cost_matrix[np.isinf(cost_matrix)] = 1e8  # 替换无穷大值为一个较大的常数
    
    # 使用 linear_sum_assignment 进行匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = {}
    for rna_idx, atac_idx in zip(row_ind, col_ind):
        # 只添加那些原本不是无穷大的匹配
        if not np.isinf(cost_backup[rna_idx, atac_idx]):
            matched_pairs[rna_ct.obs_names[rna_idx]] = atac_ct.obs_names[atac_idx]
        
    # 保存匹配的 RNA 和 ATAC 细胞
    rna_matched = [rna_cell for rna_cell in matched_pairs.keys()]
    atac_matched = [atac_cell for atac_cell in matched_pairs.values()]
    

    # print(f"********results of {ct}_{sample}********")
    # print("RNA cells: ", len(rna_ct),"||", "ATAC cells: ", len(atac_ct))
    # print(f"Matched cells of {ct}_{sample}: {len(rna_matched)}")
    # print(f"Matched percent of {ct}_{sample}: {len(atac_matched) / min(len(rna_ct), len(atac_ct)):.4f}")
    return rna_matched, atac_matched

def iterative_matching(rna: sc.AnnData, atac: sc.AnnData, 
                      n_neighbors: int = 10, max_iter: int = 5,threshold: float = 0.99,
                      output_dir: str = "results/iterative") -> sc.AnnData:
    """迭代匹配主函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化未配对细胞集合
    unpaired_rna = set(rna.obs_names)
    unpaired_atac = set(atac.obs_names)
    # 初始化已匹配细胞计数
    paierd_rna = []
    paierd_atac = []

    all_matched = []
    
    # 预计算固定参数
    
    samples = atac.obs['people'].unique().tolist()

    start_time = time.time()
    for iteration in range(max_iter):

        print(f"\n=== Iteration {iteration+1} ===")
        total_matched = 0
        


        for sample in tqdm(samples, desc=f"Sample", leave=False):

            # 获取当前未配对的细胞
            current_rna = rna[rna.obs_names.isin(unpaired_rna)]
            current_atac = atac[atac.obs_names.isin(unpaired_atac)]

            if iteration == 0:
                #第一次迭代，两种模式结果相同
                rna_m, atac_m = process_celltype_sample(
                    rna, atac, sample, 
                    os.path.join(output_dir, f"iter{iteration}"),
                    n_neighbors)
                #更新参数
                if rna_m == 0:
                    continue
                unpaired_rna -= set(rna_m)
                unpaired_atac -= set(atac_m)
                total_matched += len(rna_m)
                save_matched_data(
                    rna[rna_m],
                    atac[atac_m],
                    os.path.join(output_dir, f"iter{iteration}/{ct}_{sample}.h5ad")
                )
            else:
                #匹配模式1：全量RNA + 未配对ATAC
                rna_m1, atac_m1 = process_celltype_sample(
                    rna, current_atac, sample,  
                    os.path.join(output_dir, f"iter{iteration}"),
                    n_neighbors)
                # 匹配模式2：未配对RNA + 全量ATAC
                rna_m2, atac_m2 = process_celltype_sample(
                    current_rna, atac, sample, 
                    os.path.join(output_dir, f"iter{iteration}"),
                    n_neighbors)
                
                #更新参数
                if rna_m1 == 0:
                    if rna_m2 == 0:
                        continue
                    else:
                        unpaired_rna -= set(rna_m2)
                        
                elif rna_m2 == 0:
                    if rna_m1 == 0:
                        continue
                    else:
                        unpaired_atac -= set(atac_m1)
                        
                else:
                    unpaired_rna -= set(rna_m2)
                    unpaired_atac -= set(atac_m1)

                if rna_m1 != 0:
                    total_matched += len(rna_m1)
                    save_matched_data(
                        rna[rna_m1], 
                        atac[atac_m1], 
                        os.path.join(output_dir, f"iter{iteration}/{ct}_{sample}_mode1.h5ad")
                    )
                if rna_m2 != 0:
                    total_matched += len(rna_m2)
                    save_matched_data(
                        rna[rna_m2], 
                        atac[atac_m2], 
                        os.path.join(output_dir, f"iter{iteration}/{ct}_{sample}_mode2.h5ad")
                    )

    
        # 判断终止条件
        if total_matched == 0:
            print("No more matches found, stopping iteration.")
            end_time = time.time()
            print(f"After iteration {iteration+1}, took {end_time - start_time:.2f} seconds. End.")

            break


        # 合并本轮所有匹配结果
        iter_adata = combine_results(os.path.join(output_dir, f"iter{iteration}"))

        print(iter_adata)
        paierd_rna += iter_adata.obs['rna_cellname'].unique().tolist()
        paierd_atac += iter_adata.obs['atac_cellname'].unique().tolist()

        all_matched.append(iter_adata)

        # 计算匹配率
        rna_matched_ratio = len(set(paierd_rna)) / rna.shape[0]
        atac_matched_ratio = len(set(paierd_atac)) / atac.shape[0]
        print(f"Iteration {iteration+1}: {rna_matched_ratio:.2%} of RNA cells and {atac_matched_ratio:.2%} of ATAC cells matched.")
 
        #iter_adata.write_h5ad(os.path.join(output_dir, f"matched_iteration_{iteration+1}_update.h5ad"))
        print(iter_adata)
        end_time = time.time()
        print(f"After iteration {iteration+1}, took {end_time - start_time:.2f} seconds.")
        if iteration > 0 and rna_matched_ratio > threshold and atac_matched_ratio > threshold:
            print("Matched_ratio exceeds threshold, stopping iteration.")
            break

    # 合并所有迭代结果
    final_adata = adata_drop_duplicates(sc.concat(all_matched))

    final_adata.write_h5ad(os.path.join(output_dir, f"{n_neighbors}_neighbors_{ct}_final_matched.h5ad"))
    
    return final_adata

def save_matched_data(rna: sc.AnnData, atac: sc.AnnData, path: str):
    """保存匹配结果"""
    ##重命名obs
    rna_obs = rna.obs.reset_index().rename(columns={'cellbarcode': 'rna_cellname'})
    atac_obs = atac.obs.reset_index().rename(columns={'index': 'atac_cellname'})
    # 重命名
    #rna_obs = rna_obs.rename(columns={'celltype_L1': 'rna_celltype_L1', 'celltype_L2': 'rna_celltype_L2', 'celltype_L3': 'rna_celltype_L3', 'sample': 'rna_samplet'})

    atac_obs = atac_obs.rename(columns={'celltype_L1': 'atac_celltype_L1','celltype_L2': 'atac_celltype_L2', 'celltype_L3': 'atac_celltype_L3', 'sample': 'atac_sample' })

    rna_obs = rna_obs.rename(columns={'sample': 'rna_sample', 'stage': 'rna_stage'})
    #atac_obs = atac_obs.rename(columns={'sample': 'atac_sample'})
    #atac_obs = atac_obs.rename(columns={'stage': 'atac_stage'})

    combined = sc.AnnData(
        X=hstack([rna.X, atac.X]),
        obs=pd.concat([rna_obs, atac_obs], axis=1),
    )

    combined.var_names = np.array(rna.var_names.tolist() + atac.var_names.tolist())
    combined.write_h5ad(path)

def combine_results(result_dir: str, delete_temp: bool = True) -> sc.AnnData:
    """合并分块结果"""
    all_files = [f for f in os.listdir(result_dir) if f.endswith(".h5ad")]
    adatas = []
    
    for f in all_files:
        adata = sc.read_h5ad(os.path.join(result_dir, f))
        adata.X = csr_matrix(adata.X)  # 确保稀疏矩阵格式
        adatas.append(adata)
    
    if delete_temp:
        shutil.rmtree(result_dir)  # 删除临时文件夹
    return sc.concat(adatas) if adatas else None

# 使用示例
if __name__ == "__main__":
    # 初始化数据
    celltypes = np.load('celltypes.npy').tolist()
    for ct in celltypes:
        print(f"********************Processing celltype: {ct}********************")
        rna, atac = init_data(
            rna_path=f"data/celltype/HIV_RNA_{ct}.h5ad",
            atac_path=f"data/celltype/HIV_ATAC_{ct}.h5ad",

        )
        
        # search best k 
        try:
            final_adata = iterative_matching(
                rna=rna,
                atac=atac,
                max_iter=50,
                threshold=0.9,
                output_dir=f"results/35_neighbors",
                n_neighbors=35
            )
            print(f"Final matched data for celltype {ct} saved to {final_adata}", len(final_adata))
        except:
            try:
                print(f"No matched data for {ct} in 35 neighbors, try 5 neighbors")
                final_adata = iterative_matching(
                    rna=rna,
                    atac=atac,
                    max_iter=50,
                    threshold=0.9,
                    output_dir=f"results/35_neighbors",
                    n_neighbors=5
                )
                print(f"Final matched data for celltype {ct} saved to {final_adata}", len(final_adata))
            except:
                print(f"skip {ct}")
            

    adata_all = combine_results(result_dir="results/35_neighbors", delete_temp=False)
    adata_all = adata_drop_duplicates(adata_all)
    adata_all.write_h5ad("results/HIV_30_neighbors_final_matched.h5ad")
    print(adata_all)

# nohup python baseline_sample_ex.py > match_celltype_atac.txt 2>&1 &

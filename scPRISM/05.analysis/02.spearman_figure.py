### 以CD74为例子

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
from scipy import stats
import os
import pickle
from matplotlib.colors import Normalize
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import scanpy as sc
from scipy.stats import gaussian_kde

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")
    

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

pkl1 = load_data("/home/share/huadjyin/home/zhouxuanchi/HIV/atac_to_gene_new_data_0218/dict_gene.pkl")

# 加载 adata 对象
adata = sc.read('/home/share/huadjyin/home/zhouxuanchi/HIV/atac_to_gene_new_data_0218/data/adata_process.h5ad', backed='r')
gene_all = adata.var_names[:582].tolist()

# 将预测值小于 0 的部分替换为 0
pkl1['pred_gene'][pkl1['pred_gene'] < 0] = 0

# 初始化数组来存储相关性结果
spearman_correlations = np.zeros(582)  # 每个基因的相关性
spearman_pvalues = np.zeros(582)  # 每个基因的 p 值

# 逐基因计算 Spearman 相关性
for i in range(582):
    corr, pval = spearmanr(pkl1['label_gene'][:, i], pkl1['pred_gene'][:, i])
    spearman_correlations[i] = corr
    spearman_pvalues[i] = pval

# 将结果保存为 DataFrame
results_df = pd.DataFrame({
    'Gene_Index': np.arange(582),
    'Spearman_Correlation': spearman_correlations,
    'P_Value': spearman_pvalues
})

# 将 Gene_Index 替换为基因符号名称
results_df['Gene_Symbol'] = [gene_all[i] for i in results_df['Gene_Index']]
results_df = results_df.drop('Gene_Index', axis=1).set_index('Gene_Symbol')

# 使用 FDR 校正 p 值
reject, pvals_corrected, _, _ = multipletests(results_df['P_Value'], method='fdr_bh')
results_df['P_Value_Corrected'] = pvals_corrected


# Set global style and parameters
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# Specify genes to plot
selected_genes = ["CD74"]

output_dir = '/home/share/huadjyin/home/lutianyu/01HIV/02data/downstream/peak2gene/individual_genes'
os.makedirs(output_dir, exist_ok=True)


for gene, row in results_df.loc[selected_genes].iterrows():
    true_values = pkl1['label_gene'][:, gene_all.index(gene)]
    pred_values = pkl1['pred_gene'][:, gene_all.index(gene)]
    
    # Create DataFrame and sample non-zero points
    df = pd.DataFrame({'True Values': true_values, 'Predicted Values': pred_values})
    non_zero_mask = df['True Values'] > 0
    zero_mask = ~non_zero_mask

    non_zero_df = df[non_zero_mask]
    zero_df = df[zero_mask]
    
    # Calculate density only for non-zero points
    xy = np.vstack([non_zero_df['True Values'], non_zero_df['Predicted Values']])
    z = gaussian_kde(xy)(xy)
    
    # Sort by density
    idx = z.argsort()
    x, y, z = (non_zero_df['True Values'].values[idx], 
               non_zero_df['Predicted Values'].values[idx], 
               z[idx])
    
    # Create figure
    plt.figure(figsize=(6, 6))
    
    # 对角线
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    plt.gca().axline((min_val, min_val), slope=1, linestyle='--', 
                     color='gray', linewidth=2, alpha=0.7, zorder=1)
    
    # 展示0值的点
    if len(zero_df) > 0:
        plt.scatter(zero_df['True Values'], zero_df['Predicted Values'], 
                   c='lightgray', s=2, alpha=0.5, zorder=2, label='Zero values')
    
    # 展示非0值的点
    scatter = plt.scatter(x, y, c=z, s=2, cmap='Spectral_r', alpha=0.7, zorder=3)
    
    # 回归线（使用所有数据点）
    sns.regplot(
        x='True Values', 
        y='Predicted Values',
        data=df,
        scatter=False,
        color='black',
        ci=95,
        line_kws={'linewidth': 2, 'alpha': 0.9, 'zorder': 4}
    )
    
    # Set plot attributes
    plt.title(f'{gene}\n(Spearman R={row["Spearman_Correlation"]:.3f}, p={row["P_Value_Corrected"]:.3e})', fontsize=12)
    plt.xlabel('True Expression', fontsize=10)
    plt.ylabel('Predicted Expression', fontsize=10)
    plt.grid(False)
    
    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save figure
    output_path = os.path.join(output_dir, f'spearman_{gene}.png')
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

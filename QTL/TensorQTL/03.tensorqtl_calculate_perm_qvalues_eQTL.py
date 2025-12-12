import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sb
import seaborn as sns
from matplotlib.pyplot import rc_context
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from scipy.sparse import issparse
import decoupler as dc
from sklearn.preprocessing import QuantileTransformer
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.preprocessing import QuantileTransformer, StandardScaler
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=100, frameon=False)
sc._settings.ScanpyConfig.n_jobs=70

import os
os.environ["R_HOME"] = "/home/yangshichen//mambaforge/envs/QTL/lib/R"
os.environ["R_LIBS_USER"] = "/home/yangshichen/mambaforge/envs/QTL/lib/R/library"
import pandas as pd
import torch
import tensorqtl
from tensorqtl import  cis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch: {torch.__version__} (CUDA {torch.version.cuda}), device: {device}")
print(f"pandas: {pd.__version__}")

#设置bed文件所在的文件夹路径
bed_dir = '/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/QTL/01.Dynamic/01.Data/02.scRNA-seq/01.pseudobulk/06.gene_expression_bed/'
output_dir = '/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/QTL/01.Dynamic/03.Result/eQTL2/'

#获取所有以.bed结尾的文件
bed_files = [f for f in os.listdir(bed_dir) if f.endswith('.bed')]

cis_df_all = pd.DataFrame()

# 遍历文件，处理并创建对应文件夹
for bed_file in bed_files:
    cell_name = os.path.splitext(bed_file)[0]  # 去掉.bed后缀
    folder_path = os.path.join(output_dir, cell_name)

    #read_expression_file and covariates
    cis_df = pd.read_csv(f'{folder_path}/all_lead_perm.csv')
    cis_df['celltype'] = cell_name
    
    #tensorqtl.calculate_qvalues
    tensorqtl.calculate_qvalues(cis_df)

    #修改名字
    cis_df = cis_df.rename(columns={"qval": "celltype_level_qval",
                                        "pval_nominal_threshold":"celltype_level_pval_nominal_threshold"})
    
    #保存
    cis_df.to_csv(f'{folder_path}/eQTL_all_lead_perm_qvalue.csv')

    #按行合并
    cis_df_all = pd.concat([cis_df_all, cis_df], axis=0, ignore_index=True)

#修改名字
cis_df_all = cis_df_all.rename(columns={"qval": "celltype_level_qval",
                                        "pval_nominal_threshold":"celltype_level_pval_nominal_threshold"})

#再次计算
tensorqtl.calculate_qvalues(cis_df_all)
cis_df_all = cis_df_all.rename(columns={"qval": "study_wise_qval",
                                        "pval_nominal_threshold": "study_wise_nominal_threshold"})

print(len(cis_df_all[cis_df_all['study_wise_qval'] < 0.05]['phenotype_id'].unique())) #2633
print(len(cis_df_all[cis_df_all['celltype_level_qval'] < 0.05]['phenotype_id'].unique())) #2974

cis_df_all.to_csv('/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/QTL/01.Dynamic/03.Result/eQTL_all_lead_perm_qvalues_2.csv')





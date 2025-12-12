import os
import pandas as pd
import subprocess
from multiprocessing import Pool
import json
from collections import defaultdict

PLINK_FILE = "/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/SAIGEQTL/input/genetics/HIV_snp"
BASE_OUTPUT_DIR = "/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/SAIGEQTL/result/"
REGION_FILE_DIR = "/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/SAIGEQTL/input/region_file_ysc"
PHENO_INPUT_DIR = "/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/SAIGEQTL/input/phenofile"
GENES_PER_BATCH = 80  # 每个步骤并行处理的基因数

with open('/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/SAIGEQTL/input/dict/gene_chr_rename.json', 'r') as f:
    gene_chr = json.load(f)
with open('/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/SAIGEQTL/input/dict/CD4+ T.json', 'r') as f:
    cell_type_genes_dict = json.load(f)

def run_step1(args):
    """
    并行执行step1 - 拟合NULL模型
    args: (cell_type, gene, pheno_file_path)
    """
    cell_type, gene, pheno_file = args
    try:
        # 创建输出目录
        cell_type_dir = os.path.join(BASE_OUTPUT_DIR, cell_type)
        gene_dir = os.path.join(cell_type_dir, gene)
        os.makedirs(gene_dir, exist_ok=True)
        
        # 第一步：拟合NULL模型
        step1_prefix = os.path.join(gene_dir, f"{gene}_null_model")
        step1_cmd = [
            "/home/yangshichen/mambaforge/envs/R/bin/Rscript", "/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/SAIGEQTL/script/step1_fitNULLGLMM_qtl.R",
            "--useSparseGRMtoFitNULL=FALSE", #没有强烈的群体结构
            "--useGRMtoFitNULL=FALSE", #没有强烈的群体结构
            f"--phenoFile={pheno_file}", #表型文件，行是细胞或样本，列包括表型（基因表达）、协变量等
            f"--phenoCol={gene}", #表型列名，就是当前要测试的基因的表达量
            "--covarColList=age,treat_time,geno_PC1,geno_PC2,pct_counts_mt,PC_1,PC_2,PC_3,PC_4,PC_5", #细胞级别的协变量
            "--sampleCovarColList=age,treat_time,geno_PC1,geno_PC2", #样本级别的协变量
            "--offsetCol=log_total_counts", #使用每个单元的总读取计数的对数作为偏移量的选项
            "--sampleIDColinphenoFile=orig.ident", #样本 ID 列
            "--cellIDColinphenoFile=cell_barcode", #细胞条形码列
            "--traitType=count", #表型是 count data (单细胞基因表达计数)
            f"--outputPrefix={step1_prefix}", #输出表头
            "--skipVarianceRatioEstimation=FALSE", #是否跳过方差比估计
            "--isRemoveZerosinPheno=FALSE", #是否在建模前去掉表型为 0 的细胞
            "--isCovariateOffset=FALSE", #协变量不是 offset，而是作为固定效应
            "--isCovariateTransform=TRUE", #对协变量进行标准化或变换
            "--skipModelFitting=FALSE", #不跳过模型拟合
            "--tol=0.00001", #模型收敛容忍度，越小模型收敛要求越严格
            f"--plinkFile={PLINK_FILE}", #基因型输入文件（plink 格式 .bed/.bim/.fam），用于后续 QTL 关联测试
            "--IsOverwriteVarianceRatioFile=TRUE" #如果已有 varianceRatio 文件，允许覆盖
        ]
        
        print(f"处理 {cell_type} - {gene}: 第一步拟合NULL模型")
        subprocess.run(step1_cmd, check=True)
        return (cell_type, gene, "step1成功")
    
    except subprocess.CalledProcessError as e:
        return (cell_type, gene, f"step1失败: {str(e)}")
    except Exception as e:
        return (cell_type, gene, f"step1其他错误: {str(e)}")

def run_step2(args):
    """
    并行执行step2 - 关联分析
    args: (cell_type, gene)
    """
    cell_type, gene = args
    try:
        # 获取当前基因的染色体
        chrom = gene_chr.get(gene)
        if not chrom:
            return (cell_type, gene, f"step2失败: 基因 {gene} 的染色体信息缺失")
        
        # 检查region文件是否存在
        gene_region_file = os.path.join(REGION_FILE_DIR, f"{gene}.txt")
        if not os.path.exists(gene_region_file):
            return (cell_type, gene, f"step2失败: region文件 {gene_region_file} 不存在")
        
        # 构建路径
        cell_type_dir = os.path.join(BASE_OUTPUT_DIR, cell_type)
        gene_dir = os.path.join(cell_type_dir, gene)
        step1_prefix = os.path.join(gene_dir, f"{gene}_null_model")
        step2_output = os.path.join(gene_dir, f"{gene}_step2_output")
        
        # 第二步：关联分析
        step2_cmd = [
            "/home/yangshichen/mambaforge/envs/R/bin/Rscript", "/media/AnalysisDisk2/Yangshichen/0_HIV_RNA/SAIGEQTL/script/step2_tests_qtl.R",
            f"--bedFile={PLINK_FILE}.bed", #基因型数据（PLINK 格式）
            f"--bimFile={PLINK_FILE}.bim", #基因型数据（PLINK 格式）
            f"--famFile={PLINK_FILE}.fam", #基因型数据（PLINK 格式）
            f"--SAIGEOutputFile={step2_output}", #结果文件输出路径
            f"--chrom={chrom}", #要分析的染色体号（通常是数字 1–22 或 X）
            "--minMAF=0.1", #过滤掉次要等位基因频率小于 10% 的 SNP
            "--minMAC=20", #过滤掉次要等位基因支持数小于 20 的 SNP
            "--LOCO=FALSE", #是否使用 Leave-One-Chromosome-Out (LOCO) 方法计算 GRM（遗传相关矩阵）
            f"--GMMATmodelFile={step1_prefix}.rda", #输入 step1 的 null model 文件（广义线性混合模型）
            "--SPAcutoff=2", #SPA (saddlepoint approximation) 的阈值，控制用于小样本稀有变异的精确校正。默认 2
            f"--varianceRatioFile={step1_prefix}.varianceRatio.txt", #输入 step1 生成的 方差比文件
            f"--rangestoIncludeFile={gene_region_file}", #指定要测试的 SNP 区域（通常是某个基因 ±1Mb 的区间）
            "--markers_per_chunk=10000" #每次计算时处理的 SNP 数量，避免内存溢出
        ]
        
        print(f"处理 {cell_type} - {gene} (chr{chrom}): 第二步关联分析")
        subprocess.run(step2_cmd, check=True)
        return (cell_type, gene, "step2成功")
    
    except subprocess.CalledProcessError as e:
        return (cell_type, gene, f"step2失败: {str(e)}")
    except Exception as e:
        return (cell_type, gene, f"step2其他错误: {str(e)}")

def process_cell_type(cell_type, genes):
    """
    处理单个细胞类型的所有基因，分两步并行执行
    """
    # 准备表型文件路径
    pheno_file = os.path.join(PHENO_INPUT_DIR, f"{cell_type}.txt")
    
    # 检查表型文件是否存在
    if not os.path.exists(pheno_file):
        print(f"\n警告: 表型文件 {pheno_file} 不存在，跳过细胞类型 {cell_type}")
        return 0, len(genes), 0, len(genes)
    
    print(f"\n开始处理细胞类型: {cell_type}")
    print(f"待处理基因数: {len(genes)}")
    
    # 第一步：并行执行所有基因的step1
    print("\n开始第一步处理...")
    step1_tasks = [(cell_type, gene, pheno_file) for gene in genes]
    step1_success = 0
    step1_failures = 0
    
    with Pool(GENES_PER_BATCH) as pool:
        for result in pool.imap_unordered(run_step1, step1_tasks):
            cell, gene, status = result
            if "成功" in status:
                step1_success += 1
            else:
                step1_failures += 1
                print(f"第一步失败: {cell} - {gene}: {status}")
    
    # 第二步：并行执行所有基因的step2（仅对step1成功的基因）
    print("\n开始第二步处理...")
    step2_tasks = [(cell_type, gene) for gene in genes]
    step2_success = 0
    step2_failures = 0
    
    with Pool(GENES_PER_BATCH) as pool:
        for result in pool.imap_unordered(run_step2, step2_tasks):
            cell, gene, status = result
            if "成功" in status:
                step2_success += 1
            else:
                step2_failures += 1
                print(f"第二步失败: {cell} - {gene}: {status}")
    
    # 统计结果
    print(f"\n细胞类型 {cell_type} 处理结果:")
    print(f"第一步成功: {step1_success}, 失败: {step1_failures}")
    print(f"第二步成功: {step2_success}, 失败: {step2_failures}")
    
    return step1_success, step1_failures, step2_success, step2_failures

##开始运行
# 检查基因染色体映射是否完整
missing_chr = [gene for genes in cell_type_genes_dict.values() 
                for gene in genes if gene not in gene_chr]
if missing_chr:
    print(f"警告: 以下基因缺少染色体信息: {', '.join(missing_chr)}")
    
# 确保基础目录存在
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
total_step1_success = 0
total_step1_failures = 0
total_step2_success = 0
total_step2_failures = 0
    
# 按顺序处理每个细胞类型
for cell_type, genes in cell_type_genes_dict.items():
    s1_success, s1_failures, s2_success, s2_failures = process_cell_type(cell_type, genes)
    total_step1_success += s1_success
    total_step1_failures += s1_failures
    total_step2_success += s2_success
    total_step2_failures += s2_failures

print("\n分析完成")
print(f"第一步总成功: {total_step1_success}, 总失败: {total_step1_failures}")
print(f"第二步总成功: {total_step2_success}, 总失败: {total_step2_failures}")

#author by yangshichen
#注意：脚本仅供参考，使用前请仔细阅读

#加载R包
library(ggplot2)
library(scales)
library(Seurat)
library(patchwork)
library(dplyr)
library(cowplot)
library(tidyr)
library(ggplot2)
library(RColorBrewer)
library(SeuratObject)
library(reshape2)
library(harmony)
library(ggpubr)
library(stringr)
library(ggrepel)
library(pheatmap)

# 设置工作路径
setwd("/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/L2")

#数据合并
object=list()
mergeFile=c()
for(i in c("HDs","INRs","IRs")){
  tmp_pvalues=read.table(paste0('/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/L2/',i,'/statistical_analysis_pvalues_',i,'.txt'),header = T,sep = "\t",
                         check.names = F)
  tmp_means=read.table(paste0('/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/L2/',i,'/statistical_analysis_means_',i,'.txt'),header = T,sep = "\t",
                       check.names = F)
  tmp=tmp_pvalues[,1:11]
  tmp_pvalues=tmp_pvalues[,12:length(tmp_pvalues)]
  tmp_means=tmp_means[,12:length(tmp_means)]
  colnames(tmp_pvalues)=paste0(colnames(tmp_pvalues),'_pvalues')
  object[[i]]=cbind(tmp,tmp_pvalues,tmp_means)
  object[[i]]$stage <- paste0(i)
  mergeFile=rbind(mergeFile,object[[i]])
}

setwd("/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/数据处理L2")
write.csv(mergeFile,"cpdb数据合并.csv",row.names = FALSE)

#数据分组（58*58）
mergeFile <- read.csv("/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/数据处理L2/cpdb数据合并.csv", 
                      header = TRUE,check.names = F)
for(i in 1:784){
  setwd("/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/数据处理L2/celltype/")
  tmp <- mergeFile[,c(2,i+11,i+11+784,1580)]
  tmp <- subset(tmp, tmp[,2] < 1)
  write.csv(tmp,paste0(colnames(tmp[3]),".csv"),row.names = FALSE)
}

#合并数据
folder_path <- "/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/数据处理L2/celltype/"
file_names <- list.files(folder_path, pattern = "\\.csv$")
data <- data.frame()
for (i in 1:length(file_names)) {
  file_path <- file.path(folder_path, file_names[i])
  temp_data <- read.csv(file_path, header = TRUE,check.names = F)
  temp_data$celltype <- colnames(temp_data[3])
  colnames(temp_data)=c('interacting_pair','pvalues','means','stage','celltype')
  data <- rbind(data, temp_data)
}

setwd("/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/数据处理L2/")
write.csv(data,"CellPhoneDB_pvalue < 1.csv")

#数据分组（配受体）
split_data <- split(data, data$interacting_pair)
for (i in seq_along(split_data)) {
  interacting_pair_name <- names(split_data)[i]
  write.csv(split_data[[i]],paste0("/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/数据处理L2/interacting_pair/",
                                   interacting_pair_name,".csv"),row.names = FALSE)
}

##作图
###1、数目热图
data <- as.matrix(read.csv("/Users/mac/Desktop/HIV/scRNA-seq/cellphoneDB/L2/HDs/count_network_HDs.csv",
                           row.names = 1,header = T,check.names = F))
diag_elems <- diag(data)
sorted_idx <- order(diag_elems) 
subdat <- data[sorted_idx, sorted_idx]
#data <- data[order(rowSums(data)), ]
#data <- data[, order(colSums(data))]
#subdat <- data[17:36,17:36]

p <- pheatmap(subdat,
              cluster_cols = F,
              cluster_rows = F,
              breaks=seq(min(subdat),max(subdat),length.out = 100),
              display_numbers = ifelse(subdat >= 90,"***",
                                       ifelse(subdat >= 80 & subdat < 90, "**", 
                                              ifelse(subdat >= 70 & subdat < 80, "*", ""))),
              number_color = c("white"),
              fontsize=8,
              color = colorRampPalette(c("#FFFFFF","#B8CFEA","#80A9D7","#5A89C4","#304A77"))(100), 
              fontsize_col = 8,
              fontsize_row = 8,
              show_colnames = T,
              cellwidth = 10, 
              cellheight = 10,
              main = "HDs Interaction Counts",
              annotation_legend	= F,
              scale="none",
              border= F)
p

setwd('/Users/mac/Desktop')
ggsave("HDs Interaction Counts.pdf",p,width=16,height=15)

























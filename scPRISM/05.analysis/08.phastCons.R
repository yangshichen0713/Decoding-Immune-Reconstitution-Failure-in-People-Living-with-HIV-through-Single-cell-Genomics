setwd("01HIV/phastCons")

library(GenomicRanges)
library(phastCons100way.UCSC.hg38)
phast <- phastCons100way.UCSC.hg38

input_file <- "peak.csv"
data <- read.csv(input_file)

# 将 Peak 列转换为 GRanges 对象
peak_ranges <- GRanges(data$Peak)

#计算保守性评分
result <- gscores(phast, peak_ranges)
conservation_scores <- result@elementMetadata@listData$default
data$ConservationScore <- conservation_scores

output_file <- "peak_conservation_score.csv"
write.csv(data, output_file, row.names = FALSE)


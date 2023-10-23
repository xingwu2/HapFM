library(optparse)
library(gpart)


PATH <- getwd()

option_list = list(
  make_option(c("-g", "--geno"), type="character", default=NULL, help="genotype matrix", metavar="character"),
  make_option(c("-s", "--snp"), type="character", default=NULL, help="snpINFO", metavar="character"),
  make_option(c("-c", "--cut"), type="double", default=NULL, help="CLQcut",metavar="float"),
  make_option(c("-o", "--out"), type="character", default=NULL, help="output prefix", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)




geno <- read.delim(opt$geno, header=T)
SNPinfo <- read.delim(opt$snp,sep = "\t", header=TRUE)
res <- BigLD(geno = geno, SNPinfo = SNPinfo,CLQmode = "density",CLQcut = opt$cut,hrstType="fast")
write.table(res,file = paste(opt$out,"_res_btmp.txt",sep=""),quote = F,sep = "\t",row.names = F)


#import modules
from sklearn import preprocessing
import subprocess
import time
import multiprocessing as mp
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import re
import sys
from sklearn.linear_model import LinearRegression

#import utility scripts

import utility_functions_g as uf
import block_partition_g as bp


parser = argparse.ArgumentParser()
parser.add_argument('-v',type = str, action= 'store',dest='vcf',help='the vcf file')
parser.add_argument('-b',type = str, action= 'store',dest='block')
parser.add_argument('-s',type = str, action= 'store',dest='size')
parser.add_argument('-r',type = float, action = 'store', dest = 'corr')
parser.add_argument('-c',type = float, action = 'store', dest = 'CLQcut')
parser.add_argument('-w',type = int, action = 'store', dest = 'window')
parser.add_argument('-hc',type = str, action = 'store', dest = 'clustering',help = "haplotype clustering method")
parser.add_argument('-y',type = str, action = 'store', dest = 'pheno',help = "phenotypic value")
parser.add_argument('-cov',type = str, action = 'store', dest = 'covariates',help = "covariates")
parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of the output files")



args = parser.parse_args()


print("program started")


######################### read chromosome size file
s = []
with open(args.size,"r") as f:
	for line in f:
		line = line.strip("\n")
		s.append(float(line))

s = np.array(s,dtype=int)
print(s)

########################## create the genotype matrix from vcf file
hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions,chromosome = uf.vcf2hapmatrix(vcf=args.vcf) # n*m matrix with n individuals and m snps

if len(s) != len(chromosome):
	sys.error("The chromsome size file has different number of chromosome than the VCF file")
else:
	for i in range(len(s)):
		print(variant_positions[chromosome[i]][-5:])
		variant_positions[chromosome[i]].append(s[i])
		print(variant_positions[chromosome[i]][-5:])

geno_matrix = {}
geno_matrix_standard = {}
IndepLD_breakpoints_index = {}
fine_breakpoints = {}
alone_SNPs_index = {}
for ch in chromosome:

	geno_matrix[ch] = np.transpose(hap_matrix_d1[ch] + hap_matrix_d2[ch])
	r,c = geno_matrix[ch].shape

	print("Read chromosome %s and %d variants" %(ch,c))

	#standardize the genotype matrix
	print("start standardizing the geno matrix of chromosome %s" %(ch))
	geno_matrix_standard[ch] = preprocessing.scale(geno_matrix[ch])

	#partition into complete independent LD blocks
	print("start finding complete independent LD blocks")

	IndepLD_breakpoints_index[ch],alone_SNPs_index[ch] = bp.CompleteLDPartition_2(standardized_genotype_matrix=geno_matrix_standard[ch],cutoff=args.corr,window_size=args.window)

	print("finish finding independent blocks in chromosome %s" %(ch),IndepLD_breakpoints_index[ch])

	print("%d complete independent LD blocks were found" %(len(IndepLD_breakpoints_index[ch])-1))

with open(args.output+"_alone_SNPs.txt", "w") as L:
	for ch in chromosome:
		for i in alone_SNPs_index[ch]:
			L.write(str(ch+"\t"+str(variant_positions[ch][i]))+"\n")

if args.block == "uniform":
	for ch in chromosome:
		fine_breakpoints[ch] = 	bp.uniform_fine_partition(IndepLD_breakpoints_index[ch],step_size = 20)

	with open(args.output+".txt", "w") as W:
		for ch in chromosome:
			for i in range(len(fine_breakpoints[ch])):
				W.write(str(ch+"\t"+'\t'.join(map(str, fine_breakpoints[ch][i]))+"\n"))

elif args.block == "plink":
	for ch in chromosome:
		fine_breakpoints[ch] = bp.plink_fine_partition(IndepLD_breakpoints_index[ch],variant_names[ch],variant_positions[ch],args.vcf,args.output,ch)
	
	with open(args.output+".txt", "w") as W:
		for ch in chromosome:
			for i in range(len(fine_breakpoints[ch])):
				W.write(str(ch+"\t"+'\t'.join(map(str, fine_breakpoints[ch][i]))+"\n"))

elif args.block == "bigld":
	for ch in chromosome:
		fine_breakpoints[ch] = bp.BigLD_fine_partition_1(IndepLD_breakpoints_index[ch],geno_matrix[ch],variant_names[ch],variant_positions[ch],args.CLQcut,args.output)
	
	with open(args.output+".txt", "w") as W:
		for ch in chromosome:
			for i in range(len(fine_breakpoints[ch])):
				W.write(str(ch+"\t"+'\t'.join(map(str, fine_breakpoints[ch][i]))+"\n"))

elif args.block == "haplospace":
	for ch in chromosome:
		before = time.time()
		fine_breakpoints[ch] = bp.haplospace_fine_partition_auto(IndepLD_breakpoints_index[ch],geno_matrix[ch],minSize=5,maxSize=300)
		after = time.time()
		print(str(after - before))

	with open(args.output+".txt", "w") as W:
		for ch in chromosome:
			for i in range(len(fine_breakpoints[ch])):
				W.write(str(ch+"\t"+'\t'.join(map(str, fine_breakpoints[ch][i]))+"\n"))
else:
	fine_breakpoints = bp.custom_fine_partition(args.block)


######################## generate the haplotype block design matrix from the hap matrix
print("start haplotype design matrix generation.")


# y = []
# with open(args.pheno,"r") as f:
# 	for line in f:
# 		line = line.strip("\n")
# 		y.append(float(line))

# C =  np.array(pd.read_csv(args.covariates,sep="\t",header=None)) 

# ## fit linear regression to calculate residuals

# reg = LinearRegression(fit_intercept=False).fit(C, y)
# prediction = reg.predict(C)
# residuals = y - prediction
# print(y[:10])
# print(prediction[:10])
# print(residuals[:10])


HaploBlock_matrix = mp.Manager().dict()
haplotype_block_name = mp.Manager().dict()
haplotype_marker_name = mp.Manager().dict()

processes = []

for ch in chromosome:
	p = mp.Process(target = uf.BlockDM_generation, args=(ch,r,hap_matrix_d1,hap_matrix_d2,geno_matrix,variant_names,variant_positions,fine_breakpoints,HaploBlock_matrix,haplotype_block_name,haplotype_marker_name,args.clustering))
	#p = mp.Process(target = uf.BlockDM_generation_cluster, args=(ch,r,hap_matrix_d1,hap_matrix_d2,geno_matrix,variant_names,variant_positions,fine_breakpoints,HaploBlock_matrix,haplotype_block_name,haplotype_marker_name,args.clustering,y))
	#p = mp.Process(target = uf.HaploDM_1_singleton, args=(ch,r,hap_matrix_d1,hap_matrix_d2,geno_matrix,variant_names,variant_positions,fine_breakpoints,HaploBlock_matrix,haplotype_block_name,args.clustering))
	#p = mp.Process(target = uf.BlockDM_generation_residual, args=(ch,r,hap_matrix_d1,hap_matrix_d2,geno_matrix,variant_names,variant_positions,fine_breakpoints,HaploBlock_matrix,haplotype_block_name,haplotype_marker_name,args.clustering,residuals))
	processes.append(p)
	p.start()

for process in processes:
	process.join()

columns = []
H = pd.DataFrame(index=range(r),columns=columns)
	
OUTPUT_BLOCK_NAMES = open(args.output+"_block_names.txt","w")
for ch in chromosome:
	for i in range(len(haplotype_block_name[ch])):
		print("%s" %(haplotype_block_name[ch][i]),file = OUTPUT_BLOCK_NAMES)

OUTPUT_MARKER_NAMES = open(args.output+"_marker_names.txt","w")
for ch in chromosome:
	for i in range(len(haplotype_marker_name[ch])):
		print("%s" %(haplotype_marker_name[ch][i]),file = OUTPUT_MARKER_NAMES)




for ch in chromosome:
	for key in HaploBlock_matrix[ch]:
		H = pd.concat([H,HaploBlock_matrix[ch][key]],axis=1,ignore_index=False)


hap_names = H.columns.values.tolist()
bimbam = uf.format_bimbam(np.transpose(H.values),hap_names)
bimbam.to_csv(args.output+".bimbam",index=False,sep=" ",header=None)

H.to_csv(args.output+"_haplotypeDM.txt",sep="\t",header=True,index=False)


print("finish constructing haplotype design matrix")


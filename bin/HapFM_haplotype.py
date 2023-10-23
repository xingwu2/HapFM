
#import modules
from sklearn import preprocessing
import subprocess
import time
import argparse
import numpy as np
import pandas as pd
import re
import sys
import os
#import utility scripts

import utility_functions as uf
import block_partition as bp

parser = argparse.ArgumentParser()
parser.add_argument('-v',type = str, action= 'store',dest='vcf',help='the vcf file')
parser.add_argument('-b',type = str, action= 'store',dest='block')
parser.add_argument('-r',type = float, action = 'store', dest = 'corr',default=0.1)
parser.add_argument('-c',type = float, action = 'store', dest = 'CLQcut',default=0.5)
parser.add_argument('-w',type = int, action = 'store', dest = 'window',default = 100)
parser.add_argument('-hc',type = str, action = 'store', dest = 'clustering',help = "haplotype clustering method",default = "modularity")
parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of the output files")

args = parser.parse_args()


print("program has started")

DIR = os.path.realpath(os.path.dirname(__file__))

########################## create the genotype matrix from vcf file
hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions,chromosome = uf.vcf2hapmatrix(vcf=args.vcf) # n*m matrix with n individuals and m snps

geno_matrix = {}
geno_matrix_standard = {}
IndepLD_breakpoints_index = {}
fine_breakpoints = {}
alone_SNPs_index = {}

geno_matrix = {}
common_geno_matrix = {}
IndepLD_common_breakpoints_index = {}
common_allele_index = {}

common_variant_names = {}
common_variant_positions = {}
alone_common_SNPs_index = {}

gw_independent_breakpoints = {}
gw_fine_breakpoints = {}
block_partitions = {}

with open(args.output+"_alone_SNPs.txt", "w") as L:
	for ch in chromosome:
		geno_matrix[ch] = np.transpose(hap_matrix_d1[ch] + hap_matrix_d2[ch])
		r,c = geno_matrix[ch].shape

		# calculating the allele frequency and find common alleles 
		allele_frequency = np.sum(geno_matrix[ch],axis = 0) / (2*r)
		common_allele_index[ch] = [i for i in range(len(allele_frequency)) if allele_frequency[i] > 0.05 and allele_frequency[i] < 0.95 ]
		common_geno_matrix[ch] = geno_matrix[ch][:,common_allele_index[ch]]
		common_variant_names[ch] = [variant_names[ch][i] for i in common_allele_index[ch]]
		common_variant_positions[ch] =[variant_positions[ch][i] for i in common_allele_index[ch]]

		#standardize the genotype matrix
		common_geno_matrix_standard = preprocessing.scale(common_geno_matrix[ch])
		#partition into complete independent LD blocks
		print("start finding complete independent LD blocks using common SNPs with maf > 0.05")
		IndepLD_common_breakpoints_index[ch],alone_common_SNPs_index[ch] = bp.CompleteLDPartition(standardized_genotype_matrix=common_geno_matrix_standard,cutoff=args.corr,window_size=args.window)
		if len(alone_common_SNPs_index[ch]) > 0:
			for i in alone_common_SNPs_index[ch]:
				full_index = common_allele_index[ch][i] # conver the common allele index to the index of all the variants 
				L.write(str(ch+"\t"+str(variant_positions[ch][full_index]))+"\n")

for ch in chromosome:
		common_breakpoints = IndepLD_common_breakpoints_index[ch]
		gw_independent_breakpoints[ch] = uf.convert_independent_genomewide_breakpoints(common_breakpoints,common_allele_index[ch],len(variant_names[ch]))
		print("%d complete independent LD blocks were found:" %(len(gw_independent_breakpoints[ch])))
		print(gw_independent_breakpoints[ch])

if args.block == "bigld":
	with open(args.output+"_fine_genomewide_partition.txt", "w") as GW_BREAKPOITS:
		for ch in chromosome:
			common_fine_breakpoints = bp.BigLD_partition(DIR,IndepLD_common_breakpoints_index[ch],common_geno_matrix[ch],common_variant_names[ch],common_variant_positions[ch],args.CLQcut,args.output)
			gw_fine_breakpoints[ch] = uf.convert_fine_genomewide_breakpoints(common_fine_breakpoints,common_allele_index[ch],len(variant_names[ch]),gw_independent_breakpoints[ch])
			for i in range(len(gw_fine_breakpoints[ch])):
				GW_BREAKPOITS.write(str(ch+"\t"+'\t'.join(map(str, gw_fine_breakpoints[ch][i]))+"\n"))	
else:
	gw_fine_breakpoints = bp.custom_fine_partition(args.block)

######################## generate the haplotype block design matrix from the hap matrix
print("start haplotype design matrix generation.")

HaploBlock_matrix = {}
haplotype_block_name = {}
haplotype_marker_name = {}


for ch in chromosome:
	HaploBlock_matrix[ch],haplotype_block_name[ch],haplotype_marker_name[ch]=uf.BlockDM_generation(ch,r,hap_matrix_d1,hap_matrix_d2,geno_matrix,variant_names,variant_positions,gw_fine_breakpoints,HaploBlock_matrix,haplotype_block_name,haplotype_marker_name,args.clustering)

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


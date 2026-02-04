
import numpy as np
import pandas as pd
import subprocess
import time
import re
import utility_functions as uf


def find_ld(i,snps,cutoff,window_size):  ###find the correlation of every SNP to a window
	n_inds,n_snps = snps.shape
	left = max(i - window_size, 0)  #define the boundary of the interval
	right = min(i + window_size,n_snps) #define the boundary of the interval

	left_snps_window = snps[:,left:(i+1)]
	right_snps_window = snps[:,i:(right+1)]
	left_cor = np.matmul(np.transpose(left_snps_window),snps[:,i])/n_inds
	left_cor_rev = np.flip(left_cor)
	right_cor = np.matmul(np.transpose(right_snps_window),snps[:,i])/n_inds
	
	left_list_ = np.where(left_cor_rev**2 > cutoff)[0]
	for j in range(len(left_list_)-1):
		if left_list_[j+1] - left_list_[j] > 10:
			left_list_ = left_list_[:j+1]
			break
	left_list_ = np.flip(left_list_) * -1

	right_list_ = np.where(right_cor**2 > cutoff)[0]
	for j in range(len(right_list_)-1):
		if right_list_[j+1] - right_list_[j] > 10:
			right_list_ = right_list_[:j+1]
			break
		
	SNPinLD_index = np.unique(np.concatenate((left_list_, right_list_))) + i
	return(SNPinLD_index)

def CompleteLDPartition(standardized_genotype_matrix,cutoff,window_size):
	
	#define variables
	n_inds,n_snps = standardized_genotype_matrix.shape
	snp_list = {}
	cummax_list = []
	max_list = []
	boundary = []

	alone_SNPs_index = []
	for i in range(n_snps):
		snp_list[i] = find_ld(i,snps=standardized_genotype_matrix,cutoff=0.1,window_size=50)
		if len(snp_list[i]) == 1:
			alone_SNPs_index.append(i)
	if len(alone_SNPs_index) > 0:
		print("QUALITY CHECK: identify %d snps that are not in LD (r2 < %f) with its 50 up/downstream neighbours." %(len(alone_SNPs_index),0.1))

	print("window_size is %d, and the correlation cutoff is %f" %(window_size, cutoff))
	for i in range(n_snps):
		snp_list[i] = find_ld(i,snps=standardized_genotype_matrix,cutoff=cutoff,window_size=window_size)

	for i in range(len(snp_list)):
		if len(snp_list[i]) > 0:
			max_list.append(np.max(snp_list[i]))
		else:
			max_list.append(i)
	
	cummax_list.append(max_list[0])
	for i in range(1,len(max_list)):
		if max_list[i] > cummax_list[i-1]:
			cummax_list.append(max_list[i])
		else:
			cummax_list.append(cummax_list[i-1])

	idx = np.where( cummax_list - np.array(range(n_snps)) == 0)
	idx = idx[0]

	boundary_ = np.concatenate(([-1],np.array(idx)))
	for i in range(len(boundary_)-1):
		left = boundary_[i]+1
		right = boundary_[i+1]
		if right - left > 0:
			boundary.append([left,right])
		else:
			boundary.append([left])

	j = 0

	while j < len(boundary):
		if len(boundary[j]) == 1:
			if j ==0:
				boundary[j+1][0] = boundary[j][0]
				del boundary[j]
			else:
				boundary[j-1][1] = boundary[j][0]
				del boundary[j]
		else:
			j += 1

	return(boundary,alone_SNPs_index)

def BigLD_partition(DIR,IndepLD_breakpoints_index,geno_matrix,variant_names,variant_positions,CLQcut,prefix,ch,maf):
	fine_breakpoints_ch = []

	#generate geno and SNPinfo for BigLD
	for I in range(len(IndepLD_breakpoints_index)):
		left = IndepLD_breakpoints_index[I][0]
		right = IndepLD_breakpoints_index[I][1]
		tmp_names = variant_names[left:right+1]
		tmp_positions = variant_positions[left:right+1]
		tmp_matrix = pd.DataFrame(geno_matrix[:,left:right+1],columns= tmp_names)
		tmp_matrix.to_csv(prefix+"_"+str(I)+"_geno_matrix"+".btmp",sep="\t",header=True,index=False)
		INFO = open(prefix+"_"+str(I)+"_snpINFO"+".btmp","w")
		INFO.write("chrN\trsID\tbp\n")
		for j in range(len(tmp_positions)):
			INFO.write(str(ch)+"\t"+str(tmp_names[j])+"\t"+str(tmp_positions[j])+"\n")
		INFO.close()

		BigLD_command = "Rscript "+DIR+"/BigLD.R -g "+prefix+"_"+str(I)+"_geno_matrix"+".btmp"+ " -s "+prefix+"_"+str(I)+"_snpINFO"+".btmp" + " -c " + str(CLQcut) + " -m " + str(maf) + " -o " + prefix+"_"+str(I)
		try:
			blocks = []
			subprocess.check_call(BigLD_command,shell=True)
			tmp_file = prefix+"_"+str(I)+"_res_btmp.txt"
			with open(tmp_file,"r") as INPUT:
				header = INPUT.readline()
				for line in INPUT:
					items = line.split("\t")
					blocks.append([variant_positions.index(int(items[5])),variant_positions.index(int(items[6]))])
			blocks[-1][1] = right

			fine_breakpoints_ch.extend(blocks)

		except subprocess.CalledProcessError as e:
			print("BigLD cannot further partition blocks in this region %i - %i" %(left,right))
			fine_breakpoints_ch.append(IndepLD_breakpoints_index[I])
		rm_command = "rm "+prefix+"_"+str(I)+"_*btmp*"
		subprocess.check_call(rm_command,shell = True)

	return(fine_breakpoints_ch)


def custom_fine_partition(block):
	genomewide_breakpoints = {}
	with open(block, "r") as f:
		for line in f:
			line = line.strip("\n")
			items = line.split("\t")
			ch = items[0]
			if ch not in genomewide_breakpoints:
				genomewide_breakpoints[ch] = [[int(items[1]),int(items[2])]]

			else:
				genomewide_breakpoints[ch].append([int(items[1]),int(items[2])])
	return(genomewide_breakpoints)


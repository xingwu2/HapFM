
import numpy as np
import pandas as pd
import subprocess
import time
import re
import utility_functions as uf


# def find_ld(i,snps,cutoff,window_size):  ###find the correlation of every SNP to a window
# 	n_inds,n_snps = snps.shape
# 	left = max(i - window_size, 0)
# 	right = min(i + window_size,n_snps)
# 	snps_window = snps[:,left:right]
# 	if left > 0:
# 		cor = np.matmul(np.transpose(snps_window),snps_window[:,window_size])/n_inds
# 		return np.where(cor**2 > cutoff) + np.array(i-window_size)
# 	else:
# 		cor = np.matmul(np.transpose(snps_window), snps_window[:,i])/n_inds


# 		return np.where(cor**2 > cutoff)

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
		if left_list_[j+1] - left_list_[j] > 20:
			left_list_ = left_list_[:j+1]
			break
	left_list_ = np.flip(left_list_) * -1

	right_list_ = np.where(right_cor**2 > cutoff)[0]
	for j in range(len(right_list_)-1):
		if right_list_[j+1] - right_list_[j] > 20:
			right_list_ = right_list_[:j+1]
			break
		
	SNPinLD_index = np.unique(np.concatenate((left_list_, right_list_))) + i
	return(SNPinLD_index)

# def CompleteLDPartition(standardized_genotype_matrix,cutoff,window_size):
	
# 	#define variables
# 	n_inds,n_snps = standardized_genotype_matrix.shape
# 	snp_list = {}
# 	cummax_list = []
# 	max_list = []
# 	print("window_size is %d, and the correlation cutoff is %f" %(window_size, cutoff))

# 	for i in range(n_snps):
# 		snp_list[i] = find_ld(i,snps=standardized_genotype_matrix,cutoff=cutoff,window_size=window_size)

# 	for i in range(len(snp_list)):
# 		if np.sum(snp_list[i]) > 0:
# 			max_list.append(np.max(snp_list[i]))
# 		else:
# 			max_list.append(i)
	
# 	cummax_list.append(max_list[0])
# 	for i in range(1,len(max_list)):
# 		if max_list[i] > cummax_list[i-1]:
# 			cummax_list.append(max_list[i])
# 		else:
# 			cummax_list.append(cummax_list[i-1])

# 	idx = np.where( cummax_list - np.array(range(n_snps)) == 0)
# 	idx = idx[0]

# 	boundary = uf.remove_duplicates(np.concatenate(([0],np.array(idx))))
	
# 	return(boundary)

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
		print("QUALITY CHECK: identify %d snps that are not in LD (r2 < %f) with its 50 up/downstream neighbours. You may consider remove these SNPs" %(len(alone_SNPs_index),0.1))

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

# def find_ld_1(i,snps,cutoff,window_size):  ###find the correlation of every SNP to a window
# 	n_inds,n_snps = snps.shape
# 	left = max(i - window_size, 0)  #define the boundary of the interval
# 	right = min(i + window_size,n_snps) #define the boundary of the interval

# 	left_snps_window = snps[:,left:(i+1)]
# 	right_snps_window = snps[:,i:(right+1)]
# 	left_cor = np.matmul(np.transpose(left_snps_window),snps[:,i])/n_inds
# 	left_cor_rev = np.flip(left_cor)
# 	right_cor = np.matmul(np.transpose(right_snps_window),snps[:,i])/n_inds
	
# 	left_list_ = np.where(left_cor_rev**2 > cutoff)[0]
# 	for j in range(len(left_list_)-1):
# 		if left_list_[j+1] - left_list_[j] > 10:
# 			left_list_ = left_list_[:j+1]
# 			break
# 	left_list_ = np.flip(left_list_) * -1

# 	right_list_ = np.where(right_cor**2 > cutoff)[0]
# 	for j in range(len(right_list_)-1):
# 		if right_list_[j+1] - right_list_[j] > 10:
# 			right_list_ = right_list_[:j+1]
# 			break
		
# 	SNPinLD_index = np.unique(np.concatenate((left_list_, right_list_))) + i
# 	return(SNPinLD_index)

# def CompleteLDPartition_1(standardized_genotype_matrix,cutoff,window_size):
	
# 	#define variables
# 	n_inds,n_snps = standardized_genotype_matrix.shape
# 	snp_list = {}
# 	cummax_list = []
# 	max_list = []


# 	print("window_size is %d, and the correlation cutoff is %f" %(window_size, cutoff))

# 	for i in range(n_snps):
# 		snp_list[i] = find_ld_1(i,snps=standardized_genotype_matrix,cutoff=cutoff,window_size=window_size)

# 	for i in range(len(snp_list)):
# 		if np.sum(snp_list[i]) > 0:
# 			max_list.append(np.max(snp_list[i]))
# 		else:
# 			max_list.append(i)
	
# 	cummax_list.append(max_list[0])
# 	for i in range(1,len(max_list)):
# 		if max_list[i] > cummax_list[i-1]:
# 			cummax_list.append(max_list[i])
# 		else:
# 			cummax_list.append(cummax_list[i-1])

# 	idx = np.where( cummax_list - np.array(range(n_snps)) == 0)
# 	idx = idx[0]

# 	boundary = np.concatenate(([0],np.array(idx)))
	
# 	return(boundary)


# def CompleteLDPartition_2(standardized_genotype_matrix,cutoff,window_size):
	
# 	#define variables
# 	n_inds,n_snps = standardized_genotype_matrix.shape
# 	snp_list = {}
# 	cummax_list = []
# 	max_list = []
# 	boundary = []

# 	print("QUALITY CHECK: identify snps that are not in LD with its neighbours. Consider remove these SNPs")
# 	alone_SNPs_index = []

# 	for i in range(n_snps):
# 		snp_list[i] = find_ld_1(i,snps=standardized_genotype_matrix,cutoff=0.1,window_size=10)
# 		if len(snp_list[i]) == 1:
# 			alone_SNPs_index.append(i)


# 	print("window_size is %d, and the correlation cutoff is %f" %(window_size, cutoff))

# 	for i in range(n_snps):
# 		snp_list[i] = find_ld_1(i,snps=standardized_genotype_matrix,cutoff=cutoff,window_size=window_size)

# 	for i in range(len(snp_list)):
# 		if len(snp_list[i]) > 0:
# 			max_list.append(np.max(snp_list[i]))
# 		else:
# 			max_list.append(i)
	
# 	cummax_list.append(max_list[0])
# 	for i in range(1,len(max_list)):
# 		if max_list[i] > cummax_list[i-1]:
# 			cummax_list.append(max_list[i])
# 		else:
# 			cummax_list.append(cummax_list[i-1])

# 	idx = np.where( cummax_list - np.array(range(n_snps)) == 0)
# 	idx = idx[0]

# 	boundary_ = np.concatenate(([-1],np.array(idx)))
# 	for i in range(len(boundary_)-1):
# 		left = boundary_[i]+1
# 		right = boundary_[i+1]
# 		if right - left > 0:
# 			boundary.append([left,right])
# 		else:
# 			boundary.append([left])
# 	return(boundary,alone_SNPs_index)

def uniform_fine_partition(IndepLD_breakpoints_index,step_size):
	fine_breakpoints_ch = {}

	for I in range(len(IndepLD_breakpoints_index)):
		if len(IndepLD_breakpoints_index[I]) == 1:
			fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
		else:
			#for blocks less than 5 snp do not perform fine block partition
			if IndepLD_breakpoints_index[I][1] - IndepLD_breakpoints_index[I][0] < 5:
				fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
			else:
				boundary_ = []
				start = IndepLD_breakpoints_index[I][0]
				end = IndepLD_breakpoints_index[I][1]

				j = step_size + start -1
				while j < end:
					boundary_.append([start,j])
					start = j + 1
					j = start + step_size - 1
				if start == end:
					boundary_.append([end])
				elif start < end:
					boundary_.append([start,end])
				else:
					sys.out("ERROR: incorrect uniform partition")

				fine_breakpoints_ch[I] = boundary_


	return(fine_breakpoints_ch)


# def plink_fine_partition(IndepLD_breakpoints_index,variant_names,variant_positions,VCF,prefix,ch):

# 	fine_breakpoints_ch = {}

# 	#generate geno and SNPinfo for plink
# 	for I in range(len(IndepLD_breakpoints_index)):

# 		if len(IndepLD_breakpoints_index[I]) == 1:
# 			fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
# 		else:
# 			#for blocks less than 5 snp do not perform fine block partition
# 			if IndepLD_breakpoints_index[I][1] - IndepLD_breakpoints_index[I][0] < 5:
# 				fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
			
# 			else:
# 				left = IndepLD_breakpoints_index[I][0]
# 				right = IndepLD_breakpoints_index[I][1]

# 				TEMP = open(prefix+"_Indep_ID.ptmp"+str(I),"w")
# 				for j in range(left,right+1):
# 					TEMP.write(variant_names[j]+"\n")
# 				TEMP.close()


# 				plink_command = "plink --vcf "+ VCF +" --blocks 'no-pheno-req' 'no-small-max-span' --blocks-max-kb 1000 --blocks-min-maf 0 --allow-extra-chr --chr "+ch+" --extract "+prefix+"_Indep_ID.ptmp"+str(I)+" --out "+prefix+"_ptmp_"+str(I)
# 				subprocess.check_call(plink_command,shell = True)
# 				format_command = "cat "+prefix+"_ptmp_"+str(I)+".blocks.det | sed -E \"s/\\s+/,/g\" > "+prefix+"_ptmp_"+str(I)+".blocks.det.formated"
# 				subprocess.check_call(format_command,shell = True)
# 				tmp_file = prefix+"_ptmp_"+str(I)+".blocks.det.formated"

# 				blocks = []
# 				with open(tmp_file,"r") as INPUT:
# 					header = INPUT.readline()
# 					for line in INPUT:
# 						items = line.split(",")
# 						blocks.append([variant_positions.index(int(items[2])),variant_positions.index(int(items[3]))])
				
# 				if len(blocks) == 0:
# 					boundary_ = [IndepLD_breakpoints_index[I]]
# 				else:
# 					boundary_ = [[blocks[0][0],blocks[0][1]]]		
						
# 					#consider the gap between boundarys
# 					for i in range(len(blocks)-1):
# 						if blocks[i+1][0] - blocks[i][1] > 1:
# 							m = int((blocks[i+1][0] - blocks[i][1] -1 ) / 5)
# 							n = (blocks[i+1][0] - blocks[i][1] - 1) % 5
								
# 							if m == 0 and n != 0:
# 								for j in range(n):
# 									boundary_.append([blocks[i][1]+j+1])

# 							elif m != 0 and n == 0:
# 								for j in range(m):
# 									boundary_.append([blocks[i][1]+1+5*j,blocks[i][1]+5+5*j])

# 							else:
# 								for j in range(m):
# 									boundary_.append([blocks[i][1]+1+5*j,blocks[i][1]+5+5*j])

# 								for k in range(n):
# 									boundary_.append([blocks[i][1]+5+5*(m-1)+1+k])
# 							boundary_.append([blocks[i+1][0],blocks[i+1][1]])
							
# 						else:
# 							boundary_.append([blocks[i+1][0],blocks[i+1][1]])

# 						# consider the left and right boundary of the block

# 					if boundary_[0][0] - left > 0:

# 						left_boundary = []
# 						m = int((boundary_[0][0] - left ) / 5)
# 						n = (boundary_[0][0] - left) % 5
								
# 						if m == 0 and n != 0:
# 							for j in range(n):
# 								left_boundary.append([left+j])
# 						elif m != 0 and n == 0:
# 							for j in range(m):
# 								left_boundary.append([left+5*j,left+4+5*j])
# 						else:
# 							for j in range(m):
# 								left_boundary.append([left+5*j,left+4+5*j])
# 							for k in range(n):
# 								left_boundary.append([left+4+5*(m-1)+1+k])
						
# 						boundary_ = left_boundary + boundary_
					
# 					if right - boundary_[-1][1] > 0:

# 						right_boundary = []

# 						m = int((right - boundary_[-1][1] ) / 5)
# 						n = (right - boundary_[-1][1] ) % 5
								
# 						if m == 0 and n != 0:
# 							for j in range(n):
# 								right_boundary.append([boundary_[-1][1]+j+1])
# 						elif m != 0 and n == 0:
# 							for j in range(m):
# 								right_boundary.append([boundary_[-1][1]+1+5*j,boundary_[-1][1]+5+5*j])
# 						else:
# 							for j in range(m):
# 								right_boundary.append([boundary_[-1][1]+1+5*j,boundary_[-1][1]+5+5*j])
# 							for k in range(n):
# 								right_boundary.append([boundary_[-1][1]+5+5*(m-1)+1+k])
# 						boundary_ = boundary_ + right_boundary
				
# 				fine_breakpoints_ch[I] = boundary_

# 				rm_command = "rm "+prefix+"_"+"*ptmp*"
# 				subprocess.check_call(rm_command,shell = True)

# 	return(fine_breakpoints_ch)

# def BigLD_fine_partition(IndepLD_breakpoints_index,geno_matrix,variant_names,variant_positions,prefix):
# 	fine_breakpoints_ch = {}

# 	#generate geno and SNPinfo for BigLD
# 	for i in range(len(IndepLD_breakpoints_index)-1):
# 		fine_breakpoints_ch[i] = [IndepLD_breakpoints_index[i]]

# 		tmp_names = variant_names[IndepLD_breakpoints_index[i]:IndepLD_breakpoints_index[i+1]+1]
# 		tmp_positions = variant_positions[IndepLD_breakpoints_index[i]:IndepLD_breakpoints_index[i+1]+1]
# 		tmp_matrix = pd.DataFrame(geno_matrix[:,IndepLD_breakpoints_index[i]:IndepLD_breakpoints_index[i+1]+1],columns= tmp_names)
# 		tmp_matrix.to_csv(prefix+"_"+str(i)+"_geno_matrix"+".btmp",sep="\t",header=True,index=False)
# 		INFO = open(prefix+"_"+str(i)+"_snpINFO"+".btmp","w")
# 		INFO.write("chrN\trsID\tbp\n")
# 		for j in range(len(tmp_positions)):
# 			INFO.write(str(1)+"\t"+str(tmp_names[j])+"\t"+str(tmp_positions[j])+"\n")
# 		INFO.close()

# 		BigLD_command = "/ysm-gpfs/pi/dellaporta/PublicData/tomato_gwas/GWAS/haplotype_finemapping/sourcecode/BigLD.R -g "+prefix+"_"+str(i)+"_geno_matrix"+".btmp"+ " -s "+prefix+"_"+str(i)+"_snpINFO"+".btmp" + " -o " + prefix+"_"+str(i)
# 		try:
# 			process = subprocess.check_call(BigLD_command,shell=True)
# 			tmp_file = prefix+"_"+str(i)+"_res_btmp.txt"
# 			with open(tmp_file,"r") as INPUT:
# 				header = INPUT.readline()
# 				for line in INPUT:
# 					items = line.split("\t")
# 					fine_breakpoints_ch[i].append(variant_positions.index(int(items[6])))
# 		except subprocess.CalledProcessError as e:
# 			print("BigLD cannot further partition blocks in this region %i - %i" %(IndepLD_breakpoints_index[i],IndepLD_breakpoints_index[i+1]))
			
# 		fine_breakpoints_ch[i].append(IndepLD_breakpoints_index[i+1])
# 		fine_breakpoints_ch[i] = uf.remove_duplicates(fine_breakpoints_ch[i])
# 		rm_command = "rm "+prefix+"_"+str(i)+"_*btmp*"
# 		subprocess.check_call(rm_command,shell = True)

# 	return(fine_breakpoints_ch)


# def BigLD_fine_partition_1(IndepLD_breakpoints_index,geno_matrix,variant_names,variant_positions,CLQcut,prefix):
# 	fine_breakpoints_ch = {}

# 	#generate geno and SNPinfo for BigLD
# 	for I in range(len(IndepLD_breakpoints_index)):

# 		if len(IndepLD_breakpoints_index[I]) == 1:
# 			fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
# 		else:
# 			#for blocks less than 5 snp do not perform fine block partition
# 			if IndepLD_breakpoints_index[I][1] - IndepLD_breakpoints_index[I][0] < 5:
# 				fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
			
# 			else:
# 				left = IndepLD_breakpoints_index[I][0]
# 				right = IndepLD_breakpoints_index[I][1]

# 				tmp_names = variant_names[left:right+1]
# 				tmp_positions = variant_positions[left:right+1]
# 				tmp_matrix = pd.DataFrame(geno_matrix[:,left:right+1],columns= tmp_names)
# 				tmp_matrix.to_csv(prefix+"_"+str(I)+"_geno_matrix"+".btmp",sep="\t",header=True,index=False)
# 				INFO = open(prefix+"_"+str(I)+"_snpINFO"+".btmp","w")
# 				INFO.write("chrN\trsID\tbp\n")
# 				for j in range(len(tmp_positions)):
# 					INFO.write(str(1)+"\t"+str(tmp_names[j])+"\t"+str(tmp_positions[j])+"\n")
# 				INFO.close()

# 				BigLD_command = "/gpfs/gibbs/pi/dellaporta/PublicData/tomato_gwas/GWAS/haplotype_finemapping/sourcecode/BigLD.R -g "+prefix+"_"+str(I)+"_geno_matrix"+".btmp"+ " -s "+prefix+"_"+str(I)+"_snpINFO"+".btmp" + " -c " + str(CLQcut) + " -o " + prefix+"_"+str(I)
# 				print(BigLD_command)
# 				try:
# 					blocks = []
# 					process = subprocess.check_call(BigLD_command,shell=True)
# 					tmp_file = prefix+"_"+str(I)+"_res_btmp.txt"
# 					with open(tmp_file,"r") as INPUT:
# 						header = INPUT.readline()
# 						for line in INPUT:
# 							items = line.split("\t")
# 							blocks.append([variant_positions.index(int(items[5])),variant_positions.index(int(items[6]))])
# 					boundary_ = [[blocks[0][0],blocks[0][1]]]			
# 					#consider the gap between boundarys
# 					for i in range(len(blocks)-1):
# 						if blocks[i+1][0] - blocks[i][1] > 1:
# 							m = int((blocks[i+1][0] - blocks[i][1] -1 ) / 5)
# 							n = (blocks[i+1][0] - blocks[i][1] - 1) % 5
							
# 							if m == 0 and n != 0:
# 								for j in range(n):
# 									boundary_.append([blocks[i][1]+j+1])

# 							elif m != 0 and n == 0:
# 								for j in range(m):
# 									boundary_.append([blocks[i][1]+1+5*j,blocks[i][1]+5+5*j])

# 							else:
# 								for j in range(m):
# 									boundary_.append([blocks[i][1]+1+5*j,blocks[i][1]+5+5*j])

# 								for k in range(n):
# 									boundary_.append([blocks[i][1]+5+5*(m-1)+1+k])
# 							boundary_.append([blocks[i+1][0],blocks[i+1][1]])
# 						else:
# 							boundary_.append([blocks[i+1][0],blocks[i+1][1]])

# 					# consider the left and right boundary of the block

# 					if boundary_[0][0] - left > 0:

# 						left_boundary = []

# 						m = int((boundary_[0][0] - left ) / 5)
# 						n = (boundary_[0][0] - left) % 5
							
# 						if m == 0 and n != 0:
# 							for j in range(n):
# 								left_boundary.append([left+j])
# 						elif m != 0 and n == 0:
# 							for j in range(m):
# 								left_boundary.append([left+5*j,left+4+5*j])
# 						else:
# 							for j in range(m):
# 								left_boundary.append([left+5*j,left+4+5*j])
# 							for k in range(n):
# 								left_boundary.append([left+4+5*(m-1)+1+k])
# 						boundary_ = left_boundary + boundary_
					
# 					if right - boundary_[-1][1] > 0:

# 						right_boundary = []

# 						m = int((right - boundary_[-1][1] ) / 5)
# 						n = (right - boundary_[-1][1] ) % 5
							
# 						if m == 0 and n != 0:
# 							for j in range(n):
# 								right_boundary.append([boundary_[-1][1]+j+1])
# 						elif m != 0 and n == 0:
# 							for j in range(m):
# 								right_boundary.append([boundary_[-1][1]+1+5*j,boundary_[-1][1]+5+5*j])
# 						else:
# 							for j in range(m):
# 								right_boundary.append([boundary_[-1][1]+1+5*j,boundary_[-1][1]+5+5*j])
# 							for k in range(n):
# 								right_boundary.append([boundary_[-1][1]+5+5*(m-1)+1+k])
# 						boundary_ = boundary_ + right_boundary
				
# 					fine_breakpoints_ch[I] = boundary_

# 				except subprocess.CalledProcessError as e:
# 					print("BigLD cannot further partition blocks in this region %i - %i" %(left,right))
# 					fine_breakpoints_ch[I]  = [IndepLD_breakpoints_index[I]]
# 				rm_command = "rm "+prefix+"_"+str(I)+"_*btmp*"
# 				subprocess.check_call(rm_command,shell = True)

# 	return(fine_breakpoints_ch)

# def BigLD_fine_partition_2(IndepLD_breakpoints_index,geno_matrix,variant_names,variant_positions,prefix):
# 	fine_breakpoints_ch = {}

# 	#generate geno and SNPinfo for BigLD
# 	for I in range(len(IndepLD_breakpoints_index)):

# 		if len(IndepLD_breakpoints_index[I]) == 1:
# 			fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
# 		else:
# 			left = IndepLD_breakpoints_index[I][0]
# 			right = IndepLD_breakpoints_index[I][1]
# 			#for blocks less than 5 snp do not perform fine block partition
# 			if right - left < 5:
# 				fine_breakpoints_ch[I] = []
# 				for i in np.arange(left,right+1):
# 					fine_breakpoints_ch[I].append([i])
# 			else:
# 				tmp_names = variant_names[left:right+1]
# 				tmp_positions = variant_positions[left:right+1]
# 				tmp_matrix = pd.DataFrame(geno_matrix[:,left:right+1],columns= tmp_names)
# 				tmp_matrix.to_csv(prefix+"_"+str(I)+"_geno_matrix"+".btmp",sep="\t",header=True,index=False)
# 				INFO = open(prefix+"_"+str(I)+"_snpINFO"+".btmp","w")
# 				INFO.write("chrN\trsID\tbp\n")
# 				for j in range(len(tmp_positions)):
# 					INFO.write(str(1)+"\t"+str(tmp_names[j])+"\t"+str(tmp_positions[j])+"\n")
# 				INFO.close()

# 				BigLD_command = "/gpfs/gibbs/pi/dellaporta/PublicData/tomato_gwas/GWAS/haplotype_finemapping/sourcecode/BigLD.R -g "+prefix+"_"+str(I)+"_geno_matrix"+".btmp"+ " -s "+prefix+"_"+str(I)+"_snpINFO"+".btmp" + " -o " + prefix+"_"+str(I)
				
# 				try:
# 					blocks = []
# 					process = subprocess.check_call(BigLD_command,shell=True)
# 					tmp_file = prefix+"_"+str(I)+"_res_btmp.txt"
# 					with open(tmp_file,"r") as INPUT:
# 						header = INPUT.readline()
# 						for line in INPUT:
# 							items = line.split("\t")
# 							blocks.append([variant_positions.index(int(items[5])),variant_positions.index(int(items[6]))])
# 					boundary_ = [[blocks[0][0],blocks[0][1]]]			
# 					#consider the gap between boundarys
# 					for i in range(len(blocks)-1):
# 						if blocks[i+1][0] - blocks[i][1] > 1:
# 							m = int((blocks[i+1][0] - blocks[i][1] -1 ) / 5)
# 							n = (blocks[i+1][0] - blocks[i][1] - 1) % 5
							
# 							if m == 0 and n != 0:
# 								for j in range(n):
# 									boundary_.append([blocks[i][1]+j+1])

# 							elif m != 0 and n == 0:
# 								for j in range(m):
# 									boundary_.append([blocks[i][1]+1+5*j,blocks[i][1]+5+5*j])

# 							else:
# 								for j in range(m):
# 									boundary_.append([blocks[i][1]+1+5*j,blocks[i][1]+5+5*j])

# 								for k in range(n):
# 									boundary_.append([blocks[i][1]+5+5*(m-1)+1+k])
# 							boundary_.append([blocks[i+1][0],blocks[i+1][1]])
# 						else:
# 							boundary_.append([blocks[i+1][0],blocks[i+1][1]])

# 					# consider the left and right boundary of the block

# 					if boundary_[0][0] - left > 1:

# 						left_boundary = []

# 						m = int((boundary_[0][0] - left ) / 5)
# 						n = (boundary_[0][0] - left) % 5
							
# 						if m == 0 and n != 0:
# 							for j in range(n):
# 								left_boundary.append([left+j])
# 						elif m != 0 and n == 0:
# 							for j in range(m):
# 								left_boundary.append([left+5*j,left+4+5*j])
# 						else:
# 							for j in range(m):
# 								left_boundary.append([left+5*j,left+4+5*j])
# 							for k in range(n):
# 								left_boundary.append([left+4+5*(m-1)+1+k])
# 						boundary_ = left_boundary + boundary_
# 					if right - boundary_[-1][1] > 1:

# 						right_boundary = []

# 						m = int((right - boundary_[-1][1] -1 ) / 5)
# 						n = (right - boundary_[-1][1] -1 ) % 5
							
# 						if m == 0 and n != 0:
# 							for j in range(n):
# 								right_boundary.append([boundary_[-1][1]+j+1])
# 						elif m != 0 and n == 0:
# 							for j in range(m):
# 								right_boundary.append([boundary_[-1][1]+1+5*j,boundary_[-1][1]+5+5*j])
# 						else:
# 							for j in range(m):
# 								right_boundary.append([boundary_[-1][1]+1+5*j,boundary_[-1][1]+5+5*j])
# 							for k in range(n):
# 								right_boundary.append([boundary_[-1][1]+5+5*(m-1)+1+k])
# 						boundary_ = boundary_ + right_boundary
				
# 					fine_breakpoints_ch[I] = boundary_

# 				except subprocess.CalledProcessError as e:
# 					print("BigLD cannot further partition blocks in this region %i - %i" %(left,right))
# 					fine_breakpoints_ch[I]  = [IndepLD_breakpoints_index[I]]
# 				rm_command = "rm "+prefix+"_"+str(I)+"_*btmp*"
# 				subprocess.check_call(rm_command,shell = True)

# 	return(fine_breakpoints_ch)

def BigLD_partition(DIR,IndepLD_breakpoints_index,geno_matrix,variant_names,variant_positions,CLQcut,prefix):
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
			INFO.write(str(1)+"\t"+str(tmp_names[j])+"\t"+str(tmp_positions[j])+"\n")
		INFO.close()

		BigLD_command = "Rscript "+DIR+"/BigLD.R -g "+prefix+"_"+str(I)+"_geno_matrix"+".btmp"+ " -s "+prefix+"_"+str(I)+"_snpINFO"+".btmp" + " -c " + str(CLQcut) + " -o " + prefix+"_"+str(I)
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


# def haplospace_fine_partition(IndepLD_breakpoints_index,geno_matrix,error_rate,minSize,maxSize):
# 	fine_breakpoints_ch = {}
# 	for i in range(len(IndepLD_breakpoints_index)):

# 		if len(IndepLD_breakpoints_index[i]) == 1:
# 			fine_breakpoints_ch[i] = [IndepLD_breakpoints_index[i]]
# 		else:
# 			#for blocks less than 5 snp do not perform fine block partition
# 			if IndepLD_breakpoints_index[i][1] - IndepLD_breakpoints_index[i][0] < 5:
# 				fine_breakpoints_ch[i] = [IndepLD_breakpoints_index[i]]

# 			else:
# 				fine_breakpoints_ch[i] = []
# 				before = time.time()
# 				tmp_matrix = geno_matrix[:,IndepLD_breakpoints_index[i][0]:IndepLD_breakpoints_index[i][1]+1]
# 				T, f = hs.HaploSpace(tmp_matrix,error_rate,minSize,maxSize)
# 				I, J = hs.bac_tra(T,minSize,maxSize)
# 				I = np.flip(I) + IndepLD_breakpoints_index[i][0]
# 				J = np.flip(J) + IndepLD_breakpoints_index[i][0]
# 				for j in range(len(I)):
# 					fine_breakpoints_ch[i].append([I[j],J[j]])
# 				#_fine_breakpoints = np.concatenate(([0],J))
# 				#fine_breakpoints_ch[i] = np.sort(np.array(_fine_breakpoints)+IndepLD_breakpoints_index[i])
# 				after = time.time()
# 				print(i,str(after - before))
# 		print(fine_breakpoints_ch[i])
# 	return(fine_breakpoints_ch)

# def haplospace_fine_partition_auto(IndepLD_breakpoints_index,geno_matrix,minSize,maxSize):
# 	fine_breakpoints_ch = {}
# 	for i in range(len(IndepLD_breakpoints_index)):

# 		if len(IndepLD_breakpoints_index[i]) == 1:
# 			fine_breakpoints_ch[i] = [IndepLD_breakpoints_index[i]]
# 		else:
# 			#for blocks less than 5 snp do not perform fine block partition
# 			if IndepLD_breakpoints_index[i][1] - IndepLD_breakpoints_index[i][0] < 5:
# 				fine_breakpoints_ch[i] = [IndepLD_breakpoints_index[i]]

# 			else:
# 				fine_breakpoints_ch[i] = []
# 				before = time.time()
# 				tmp_matrix = geno_matrix[:,IndepLD_breakpoints_index[i][0]:IndepLD_breakpoints_index[i][1]+1]
# 				T, f = hs_auto.HaploSpace(tmp_matrix,minSize,maxSize)
# 				I, J = hs_auto.bac_tra(T,minSize,maxSize)
# 				I = np.flip(I) + IndepLD_breakpoints_index[i][0]
# 				J = np.flip(J) + IndepLD_breakpoints_index[i][0]
# 				for j in range(len(I)):
# 					fine_breakpoints_ch[i].append([I[j],J[j]])
# 				#_fine_breakpoints = np.concatenate(([0],J))
# 				#fine_breakpoints_ch[i] = np.sort(np.array(_fine_breakpoints)+IndepLD_breakpoints_index[i])
# 				after = time.time()
# 				print(i,str(after - before))
# 		print(fine_breakpoints_ch[i])
# 	return(fine_breakpoints_ch)

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


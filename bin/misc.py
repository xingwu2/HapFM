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


# def uniform_fine_partition(IndepLD_breakpoints_index,step_size):
# 	fine_breakpoints_ch = {}

# 	for I in range(len(IndepLD_breakpoints_index)):
# 		if len(IndepLD_breakpoints_index[I]) == 1:
# 			fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
# 		else:
# 			#for blocks less than 5 snp do not perform fine block partition
# 			if IndepLD_breakpoints_index[I][1] - IndepLD_breakpoints_index[I][0] < 5:
# 				fine_breakpoints_ch[I] = [IndepLD_breakpoints_index[I]]
# 			else:
# 				boundary_ = []
# 				start = IndepLD_breakpoints_index[I][0]
# 				end = IndepLD_breakpoints_index[I][1]

# 				j = step_size + start -1
# 				while j < end:
# 					boundary_.append([start,j])
# 					start = j + 1
# 					j = start + step_size - 1
# 				if start == end:
# 					boundary_.append([end])
# 				elif start < end:
# 					boundary_.append([start,end])
# 				else:
# 					sys.out("ERROR: incorrect uniform partition")

# 				fine_breakpoints_ch[I] = boundary_


# 	return(fine_breakpoints_ch)

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



'''
define functions
'''

''' 

preprocessing steps:

1. convert vcf to SNP matrix (sum of two haplotype matrix)
2. LD measurement
3. independent LD block partition  

'''

# def vcf2hapmatrix(vcf):
# 	hap_matrix_d1 = {} #haplotype 1 of individuals, key as chromosome number
# 	hap_matrix_d2 = {} #haplotype 2 of individuals, key as chromosome number
# 	variant_names = {}
# 	variant_positions = {} #key as chromosome number
# 	chromosome = [] #key as chromosome number and value as number of SNPs per chromosome
	
# 	with open(vcf,"r") as VCF:
# 		for line in VCF:
# 			if re.search("^##",line): ## skip the first annotation lines
# 				continue
# 			elif re.search("^#CHROM",line): ## acquire the sample name information
# 				line = line.strip("\n")
# 				ind_names = line.split("\t")[9:]
# 			else:
# 				line = line.strip("\n")
# 				items = line.split("\t")
# 				ch = items[0]

# 				if ch not in chromosome:
# 					chromosome.append(ch)
# 					variant_names[ch] = [items[2]]
# 					variant_positions[ch] = [int(items[1])]
# 					hap_matrix_d1[ch] = []
# 					hap_matrix_d2[ch] = []
# 					genotype = items[9:]
# 					for i in range(len(genotype)):
# 						m = re.search('([0-9])\|([0-9])',genotype[i])
# 						hap_matrix_d1[ch].append(int(m.group(1)))
# 						hap_matrix_d2[ch].append(int(m.group(2)))
# 				else:
# 					variant_names[ch].append(items[2])
# 					variant_positions[ch].append(int(items[1]))
# 					genotype = items[9:]
# 					for i in range(len(genotype)):
# 						m = re.search('([0-9])\|([0-9])',genotype[i])
# 						hap_matrix_d1[ch].append(int(m.group(1)))
# 						hap_matrix_d2[ch].append(int(m.group(2)))

# 	for ch in chromosome:
# 		hap_matrix_d1[ch] = np.reshape(np.asarray(hap_matrix_d1[ch],dtype=int),(len(variant_names[ch]),len(ind_names)))
# 		hap_matrix_d2[ch] = np.reshape(np.asarray(hap_matrix_d2[ch],dtype=int),(len(variant_names[ch]),len(ind_names)))

# 	return(hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions,chromosome)

# def pip_calculation_1(haplotype_burnt_gamma,block_haplotypes,block_positions):

# 	nrow = haplotype_burnt_gamma.shape[0]
# 	ncol = len(block_haplotypes)
# 	block_gamma = np.zeros(shape = (nrow,ncol))
# 	for i in range(len(block_positions)):
# 		col_index = block_haplotypes[block_positions[i]]
# 		x = np.sum(haplotype_burnt_gamma[:,col_index],axis = 1)
# 		row_index = np.where(x >= 1)
# 		block_gamma[row_index[0],i] = 1
# 	block_pip = np.mean(block_gamma,axis = 0)
# 	return(block_pip)


# def sampling_w_annotation(y,C,HapDM,annotation,sig0_initiate,sig1_initiate,sige_initiate,pie_initiate,step_size,iters,prefix):

# 	#initiate beta,gamma and H matrix
# 	H_r,H_c = H.shape

# 	##specify hyper parameters
# 	pie_a = 1
# 	pie_b = H_c / 10
# 	a_sigma = 1
# 	b_sigma = 1
# 	a_e = 1
# 	b_e = 1

# 	sigma_0 = sig0_initiate
# 	sigma_1 = sig1_initiate
# 	sigma_e = sige_initiate
# 	pie = pie_initiate
	
# 	print("initiate:",sigma_1,sigma_e,pie)

# 	H = np.array(HapDM)

# 	#initiate alpha, alpha_trace, beta_trace and gamma_trace

# 	it = 0
# 	burn_in_iter = 2000
# 	trace = np.empty((iters-2000,6))
# 	alpha_trace = np.empty((iters-2000,C_c))
# 	theta_trace = np.empty((iters-2000,annotation.shape[1]))
# 	gamma_trace = np.empty((iters-2000,H_c))
# 	beta_trace = np.empty((iters-2000,H_c))
# 	top5_beta_trace = np.empty((iters-2000,5))


# 	alpha = np.random.random(size = C_c)

# 	theta = np.append(sp.stats.norm.ppf(pie_initiate),np.repeat(0,annotation.shape[1]-1))
	
# 	Z = np.matmul(annotation,theta)
# 	pie = sp.stats.norm.cdf(Z)

# 	gamma = np.random.binomial(1,pie_initiate,H_c)
	
# 	beta = np.array(np.zeros(H_c))
# 	for i in range(H_c):
# 		if gamma[i] == 0:
# 			beta[i] = np.random.normal(0,sigma_0)
# 		else:
# 			beta[i] = np.random.normal(0,sigma_1) 


# 	H_beta = np.matmul(H,beta)
# 	C_alpha = np.matmul(C,alpha)
# 	#start sampling

# 	while it < iters:
# 		before = time.time()
# 		Z = sample_Z(theta,gamma,annotation)
# 		theta = sample_theta(annotation,Z)
# 		gamma = sample_gamma_annotation(beta,gamma,sigma_0,sigma_1,annotation,theta)
# 		sigma_1 = sample_sigma_1(beta,gamma,a_sigma,b_sigma)
# 		sigma_e = sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
# 		alpha,C_alpha = sample_alpha(y,H_beta,C_alpha,C,alpha,sigma_e)
# 		beta,H_beta = sample_beta(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e)
# 		after = time.time()
# 		genetic_var = np.var(H_beta)
		
# 		pheno_var = np.var(y - C_alpha)
# 		large_beta = np.absolute(beta) > 0.3
# 		large_beta_ratio = np.sum(large_beta) / len(beta)
# 		large_pie = sp.stats.norm.cdf(Z_update) > 0.1
# 		large_pie_ratio = np.sum(large_pie) / len(Z_update)
# 		total_heritability = genetic_var / pheno_var


# 		if it > 100 and  total_heritability > 1:
# 			print("unrealistic beta sample",genetic_var,pheno_var)
# 			continue
# 		else:
# 			if it >= burn_in_iter:
# 				trace[it-burn_in_iter,:] = [it,sigma_1,sigma_e,large_beta_ratio,large_pie_ratio,total_heritability]
# 				gamma_trace[it-burn_in_iter,:] = gamma
# 				beta_trace[it-burn_in_iter,:] = beta
# 				alpha_trace[it-burn_in_iter,:] = alpha
# 				theta_trace[it-burn_in_iter,:] = theta
# 				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

# 			if it >= burn_in_iter + 9999: # after burn-in iterations, test convergence

# 				max_z = []

# 				# for t in range(len(theta)):
# 				#  	after_burnin_theta = theta_trace[:,t]
# 				#  	theta_zscores = pm3.geweke(after_burnin_theta)[:,1]
# 				#  	max_z.append(np.amax(np.absolute(theta_zscores)))

# 				for a in range(C_c):
# 					after_burnin_alpha = alpha_trace[:,a]
# 					alpha_zscores = geweke.geweke(after_burnin_alpha)[:,1]
# 					max_z.append(np.amax(np.absolute(alpha_zscores)))

# 				for b in range(5):
# 					after_burnin_beta = top5_beta_trace[:,b]
# 					beta_zscores = geweke.geweke(after_burnin_beta)[:,1]
# 					max_z.append(np.amax(np.absolute(beta_zscores)))

# 				#convergence for large beta ratio
# 				after_burnin_pie = trace[:,4]
# 				pie_zscores = geweke.geweke(after_burnin_pie)[:,1]
# 				max_z.append(np.amax(np.absolute(pie_zscores)))

# 				#convergence for large pi ratio
# 				after_burnin_beta_ratio = trace[:,3]
# 				pie_zscores = geweke.geweke(after_burnin_beta_ratio)[:,1]
# 				max_z.append(np.amax(np.absolute(pie_zscores)))

# 				#convergence for total heritability
# 				after_burnin_var = trace[:,5]
# 				var_zscores = geweke.geweke(after_burnin_var)[:,1]
# 				max_z.append(np.amax(np.absolute(var_zscores)))

# 				#convergence for sigma_1
# 				after_burnin_sigma1 = trace[:,1]
# 				sigma1_zscores = geweke.geweke(after_burnin_sigma1)[:,1]
# 				max_z.append(np.amax(np.absolute(sigma1_zscores)))

# 				#convergence for sigma_e
# 				after_burnin_sigmae = trace[:,2]
# 				sigmae_zscores = geweke.geweke(after_burnin_sigmae)[:,1]
# 				max_z.append(np.amax(np.absolute(sigmae_zscores)))
				
# 				if  np.amax(max_z) < 1.5:
# 					print("convergence has been reached at %i iterations." %(it))
# 					break

# 				else:
# 					trace_ = np.empty((1000,6))
# 					gamma_trace_ = np.empty((1000,H_c))
# 					beta_trace_ = np.empty((1000,H_c))
# 					alpha_trace_ = np.empty((1000,C_c))
# 					theta_trace_ = np.empty((1000,annotation.shape[1]))
# 					top5_beta_trace_ = np.empty((1000,5))

# 					trace = np.concatenate((trace[-(iters - burn_in_iter-1000):,:],trace_),axis=0)
# 					gamma_trace = np.concatenate((gamma_trace[-(iters - burn_in_iter-1000):,:],gamma_trace_),axis=0)
# 					beta_trace = np.concatenate((beta_trace[-(iters - burn_in_iter-1000):,:],beta_trace_),axis=0)
# 					alpha_trace = np.concatenate((alpha_trace[-(iters - burn_in_iter-1000):,:],alpha_trace_),axis=0)
# 					theta_trace = np.concatenate((theta_trace[-(iters - burn_in_iter-1000):,:],theta_trace_),axis=0)
# 					top5_beta_trace = np.concatenate((top5_beta_trace[-(iters - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

# 					burn_in_iter += 1000
# 					iters += 1000

# 			if (it - burn_in_iter) >= 0 and (it - burn_in_iter ) % 1000 == 0:
# 				print("%i iterations have sampled" %(it), str(after - before),trace[it-burn_in_iter,:])

# 			it += 1

# 	trace = pd.DataFrame(trace)
# 	alpha_trace = pd.DataFrame(alpha_trace)
# 	beta_trace = pd.DataFrame(beta_trace)
# 	gamma_trace = pd.DataFrame(gamma_trace)
# 	theta_trace = pd.DataFrame(theta_trace)
# 	return(trace,alpha_trace,beta_trace,gamma_trace,theta_trace)

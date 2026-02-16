#import modules
import time
import argparse
import numpy as np
import pandas as pd
import re
import multiprocessing as mp
import time

import utility_functions as uf
import gibbs_sampling as gs

def main():

	args = uf.parse_arguments_mapping()

	HapDM = pd.read_csv(args.input,sep="\t")
	n,p = HapDM.shape

	print("Successfully load the haplotypeDM matrix, found %i individuals and %i haplotypes" %(n,p))

	hap_names = HapDM.columns.values.tolist()
	HapDM = np.array(HapDM)

	y = []
	with open(args.phenotype,"r") as f:
		for line in f:
			line = line.strip("\n")
			y.append(line)
	
	NA_index = [i for i, x in enumerate(y) if x != "NA"]

	print("Finished loading the phenotype file, found %i individuals with  %i missing phenotype(s), individual(s) with missing phenotype will be removed from the following analyses" %(len(y),len(y)-len(NA_index)))


	y_noNA = [y[i] for i in NA_index]

	y = np.asarray(y_noNA,dtype=np.float64)
	#print(y)

	if args.covariates is None:
		C = np.ones(n)
		C = C.reshape(n, 1)
	else:
		C =  np.array(pd.read_csv(args.covariates,sep="\t",header=None)) 

	C = C[NA_index,:]

	HapDM = HapDM[NA_index,:]


	before = time.time()

	block_haplotypes = {}
	block_positions = []

	for i in range(len(hap_names)):
		block_name_ = re.compile("(.*@.*)_[0-9]+")
		m = block_name_.search(hap_names[i])
		if m.group(1) in block_haplotypes:
			block_haplotypes[m.group(1)].append(i)
		else:
			block_haplotypes[m.group(1)] = [i]
			block_positions.append(m.group(1))

	trace_container = mp.Manager().dict()
	gamma_container = mp.Manager().dict()
	beta_container = mp.Manager().dict()
	alpha_container = mp.Manager().dict()
	block_gamma_container = mp.Manager().dict()
	convergence_container = mp.Manager().dict()

	processes = []

	if args.mode == 1:
		for num in range(args.num):
			p = mp.Process(target = gs.sampling, args=(args.verbose,y,C,HapDM,args.s0,block_haplotypes,block_positions,num,trace_container,gamma_container,beta_container,alpha_container,block_gamma_container,convergence_container))
			processes.append(p)
			p.start()
	else:
		for num in range(args.num):
			p = mp.Process(target = gs.sampling_w_annotation, args=(y,C,HapDM,args.s0,args.s1,args.se,args.pie,args.output,num,trace_container,gamma_container,beta_container,alpha_container))
			processes.append(p)
			p.start()

	for process in processes:
		process.join()


	convergence_all_chains = []
	alpha_posterior_all_chains = []
	alpha_posterior_sd_all_chains = []
	beta_posterior_all_chains = []
	beta_posterior_sd_all_chains = []
	gamma_all_chains = []
	block_gamma_all_chains = []
	trace_posterior_all_chains = []

	column_names = ["alpha_norm_2","beta_norm_2","sigma_1","sigma_e","large_beta_ratio","total_heritability","sum_gamma"]

	for num in range(args.num):
		convergence_all_chains.append(convergence_container[num])

	print("%i/%i chains have reached the convergence." %(np.sum(convergence_all_chains),len(convergence_all_chains)))

	if np.sum(convergence_all_chains) > 0:
		for num in range(args.num):
			if convergence_all_chains[num] == 1:
				alpha_posterior_all_chains.append(alpha_container[num]["avg"])
				alpha_posterior_sd_all_chains.append(alpha_container[num]["M2"])
				beta_posterior_all_chains.append(beta_container[num]["avg"])
				beta_posterior_sd_all_chains.append(beta_container[num]["M2"])
				trace_posterior_all_chains.append(trace_container[num])
				gamma_all_chains.append(gamma_container[num])
				block_gamma_all_chains.append(block_gamma_container[num])
		
		trace_posterior_all_chains = np.vstack(trace_posterior_all_chains)
		trace_posterior = np.mean(trace_posterior_all_chains,axis=0)
		trace_posterior_sd = np.std(trace_posterior_all_chains,axis=0)

		pip = np.mean(gamma_all_chains,axis=0)
		block_pip = np.mean(block_gamma_all_chains,axis=0)

		## sort blocks by fdrs
		index,fdr = uf.fdr_calculation(block_pip)
\
		block_positions_sorted = [block_positions[i] for i in index]
		block_pip_sorted = [block_pip[i] for i in index]

		beta_posterior = []
		beta_posterior_M2 = []
		alpha_posterior = []
		alpha_posterior_M2 = []
				
		N_beta=0
		N_alpha=0

		for num in range(args.num):
			if convergence_all_chains[num] == 1:
				beta_posterior,beta_posterior_M2,N_beta = uf.merge_welford(beta_posterior,beta_posterior_M2,N_beta,beta_container[num]["avg"],beta_container[num]["M2"],10000)
				alpha_posterior,alpha_posterior_M2,N_alpha = uf.merge_welford(alpha_posterior,alpha_posterior_M2,N_alpha,alpha_container[num]["avg"],alpha_container[num]["M2"],10000)

		beta_posterior_sd = np.sqrt(beta_posterior_M2/(N_beta-1))
		alpha_posterior_sd = np.sqrt(alpha_posterior_M2/(N_alpha-1))
		np.savetxt(args.output+"_model_trace.txt",trace_posterior_all_chains,delimiter="\t",header="\t".join(column_names))

		OUTPUT_TRACE = open(args.output+"_param.txt","w")
		for i in range(len(trace_posterior)):
			print("%s\t%f\t%f" %(column_names[i],trace_posterior[i],trace_posterior_sd[i]),file = OUTPUT_TRACE)
				
		OUTPUT_ALPHA = open(args.output+"_alpha.txt","w")
		for i in range(len(alpha_posterior)):
			print("%f\t%f" %(alpha_posterior[i],alpha_posterior_sd[i]),file = OUTPUT_ALPHA)

		OUTPUT_BETA = open(args.output+"_haplotype_beta.txt","w")
		print("haplotype_block\tbeta_mean\tbeta_sd\tpip",file = OUTPUT_BETA)
		for i in range(len(beta_posterior)):
			print("%s\t%f\t%f\t%f" %(hap_names[i],beta_posterior[i],beta_posterior_sd[i],pip[i]),file = OUTPUT_BETA)

		OUTPUT_BLOCK = open(args.output+"_block_pip.txt","w")
		print("haplotype_block\tpip\tfdr",file = OUTPUT_BLOCK)
		for i in range(len(block_positions_sorted)):
			print("%s\t%f\t%f" %(block_positions_sorted[i],block_pip_sorted[i],fdr[i]),file = OUTPUT_BLOCK)


	else:
		OUTPUT_TRACE = open(args.output+"_param.txt","w")
		for i in range(len(column_names)):
			print("%s\t%s\t%s" %(column_names[i],"NA","NA"),file = OUTPUT_TRACE)
				
		OUTPUT_ALPHA = open(args.output+"_alpha.txt","w")
		for i in range(C.shape[1]):
			print("%s\t%s" %("NA","NA"),file = OUTPUT_ALPHA)

		OUTPUT_BETA = open(args.output+"_beta.txt","w")
		print("haplotype_block\tbeta_mean\tbeta_sd\tpip",file = OUTPUT_BETA)
		for i in range(X.shape[1]):
			print("%s\t%s\t%s\t%s" %(hap_names[i],"NA","NA","NA"),file = OUTPUT_BETA)


if __name__ == "__main__":
	main()
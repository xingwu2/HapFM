#import modules
import time
import argparse
import numpy as np
import pandas as pd
import re

import utility_functions as uf
import gibbs_sampling as gs


parser = argparse.ArgumentParser()
parser.add_argument('-i',type = str, action = 'store', dest = 'input')
parser.add_argument('-c',type = str, action = 'store', dest = 'covariates')
parser.add_argument('-y',type = str, action = 'store', dest = 'phenotype')
parser.add_argument('-a',type = str, action = 'store', dest = 'annotation')
parser.add_argument('-m',type = int, action = 'store', default = 1,dest = 'mode',help = "1:no annotation; 2:with annotation file")
parser.add_argument('-s0',type = float, action = 'store', dest = 's0',default = 0.01, help = "initiation for sigma0")
parser.add_argument('-s1',type = float, action = 'store', dest = 's1',default = 1, help = "initiation for sigma1")
parser.add_argument('-se',type = float, action = 'store', dest = 'se',default = 1, help = "initiation for sigmae")
parser.add_argument('-p',type = float, action = 'store', dest = 'pie',default = 0.001, help = "initiation for pie")
parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of the output files")


args = parser.parse_args()

HapDM = pd.read_csv(args.input,sep="\t")
hap_names = HapDM.columns.values.tolist()

y = []
with open(args.phenotype,"r") as f:
	for line in f:
		line = line.strip("\n")
		y.append(float(line))

y = np.asarray(y)

C =  np.array(pd.read_csv(args.covariates,sep="\t",header=None)) 

if args.mode == 1:
	trace,alpha_trace,beta_trace,gamma_trace = gs.sampling(y,C,HapDM,args.s0,args.s1,args.se,args.pie,iters=12000,prefix=args.output)
	gamma_trace.columns = hap_names
	beta_trace.columns = hap_names
	trace.to_csv(args.output+"_trace.txt",sep="\t",header=False,index=False)

	alpha_trace_avg = np.mean(alpha_trace,axis=0)
	alpha_trace_sd = np.std(alpha_trace,axis = 0)
	OUTPUT_ALPHA = open(args.output+"_alpha.txt","w")
	for i in range(C.shape[1]):
		print("%f\t%f" %(alpha_trace_avg[i],alpha_trace_sd[i]),file = OUTPUT_ALPHA)

	beta_trace_avg = np.mean(beta_trace,axis=0)
	beta_trace_sd = np.std(beta_trace,axis = 0)
	OUTPUT_BETA = open(args.output+"_beta.txt","w")
	for i in range(len(hap_names)):
		print("%s\t%f\t%f" %(hap_names[i],beta_trace_avg[i],beta_trace_sd[i]),file = OUTPUT_BETA)


elif args.mode == 2:
	if args.annotation == None:
		sys.out("ERROR: please provide the annotation file")
	else:
		Annotation = np.array(pd.read_csv(args.annotation,sep="\t"))
		trace,alpha_trace,beta_trace,gamma_trace,theta_trace = gs.sampling_w_annotation(y,C,HapDM,Annotation,args.s0,args.s1,args.se,args.pie,step_size=args.step,iters=10000,prefix=args.output)
		gamma_trace.columns = hap_names
		beta_trace.columns = hap_names
		trace.to_csv(args.output+"_trace.txt",sep="\t",header=False,index=False)
		
		alpha_trace_avg = np.mean(alpha_trace,axis=0)
		alpha_trace_sd = np.std(alpha_trace,axis = 0)
		OUTPUT_ALPHA = open(args.output+"_alpha.txt","w")
		for i in range(C.shape[1]):
			print("%f\t%f" %(alpha_trace_avg[i],alpha_trace_sd[i]),file = OUTPUT_ALPHA)

		theta_trace_avg = np.mean(theta_trace,axis=0)
		theta_trace_sd = np.std(theta_trace,axis = 0)
		OUTPUT_THETA = open(args.output+"_alpha.txt","w")
		for i in range(Annotation.shape[1]):
			print("%f\t%f" %(theta_trace_avg[i],theta_trace_sd[i]),file = OUTPUT_THETA)

		beta_trace_avg = np.mean(beta_trace,axis=0)
		beta_trace_sd = np.std(beta_trace,axis = 0)
		OUTPUT_BETA = open(args.output+"_beta.txt","w")
		for i in range(len(hap_names)):
			print("%s\t%f\t%f" %(hap_names[i],beta_trace_avg[i],beta_trace_sd[i]),file = OUTPUT_BETA)

else:
	sys.out("ERROR: Unknown sampling mode")

################# PIP calculation



haplotype_burnt_gamma = np.array(gamma_trace)

haplotype_pip = np.mean(haplotype_burnt_gamma,axis = 0)

OUTPUT_HAP = open(args.output+"_haplotype_pip.txt","w")

for i in range(len(hap_names)):
	print("%s\t%f" %(hap_names[i],haplotype_pip[i]),file = OUTPUT_HAP)

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

block_pip_1 = uf.pip_calculation_1(haplotype_burnt_gamma,block_haplotypes,block_positions)

# block_pip_2 = uf.pip_calculation_2(haplotype_pip,block_haplotypes,block_positions)

# block_pip_max = uf.pip_calculation_max(haplotype_pip,block_haplotypes,block_positions)


OUTPUT_BLOCK_1 = open(args.output+"_block_pip.txt","w")
for i in range(len(block_pip_1)):
	print("%s\t%s" %(block_positions[i],block_pip_1[i]),file = OUTPUT_BLOCK_1)

# OUTPUT_BLOCK_2 = open(args.output+"_block_pip_2.txt","w")
# for i in range(len(block_pip_2)):
# 	print("%s\t%s" %(block_positions[i],block_pip_2[i]),file = OUTPUT_BLOCK_2)

# OUTPUT_BLOCK_MAX = open(args.output+"_block_pip_max.txt","w")
# for i in range(len(block_pip_max)):
# 	print("%s\t%s" %(block_positions[i],block_pip_max[i]),file = OUTPUT_BLOCK_MAX)



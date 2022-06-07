
import numpy as np
import scipy as sp
import math
from sklearn import preprocessing
import pandas as pd 
import time
import pymc3 as pm3

def sample_gamma_annotation(beta,gamma,sigma_0,sigma_1,A,theta):
	inv_pie = np.matmul(A,theta)
	pie = sp.stats.norm.cdf(inv_pie)
	#beta is a list of marginal beta_i
	p = np.empty(len(beta))
	d1 = np.multiply(pie,sp.stats.norm.pdf(beta,loc=0,scale=sigma_1))
	d0 = np.multiply((1-pie),sp.stats.norm.pdf(beta,loc=0,scale=sigma_0))
	p = d1/(d0+d1)
	gamma = np.random.binomial(1,p)
	return(gamma)

def sample_theta(A,Z):
	ATA_inv = np.linalg.inv(np.matmul(np.transpose(A),A))
	ATZ = np.matmul(np.transpose(A),Z)
	mean = np.matmul(ATA_inv,ATZ)
	covariance = ATA_inv
	theta = np.random.multivariate_normal(mean,covariance)
	return(theta)


def sample_Z(theta,gamma,A):
	mean = np.dot(A,theta)
	variance = 1
	Z = np.random.random_sample(len(gamma))
	for i in range(len(gamma)):
		if gamma[i] == 1:
			lower,upper = 0, np.inf
			Z[i] = sp.stats.truncnorm.rvs(
				(lower - mean[i])/math.sqrt(variance),(upper-mean[i])/math.sqrt(variance),loc=mean[i],scale=math.sqrt(variance)
				)
		else:
			lower,upper = -np.inf, 0
			Z[i] = sp.stats.truncnorm.rvs(
				(lower - mean[i])/math.sqrt(variance),(upper-mean[i])/math.sqrt(variance),loc=mean[i],scale=math.sqrt(variance)
				)
	return(Z)

def sample_gamma(beta,sigma_0,sigma_1,pie):
	p = np.empty(len(beta))
	d1 = pie*sp.stats.norm.pdf(beta,loc=0,scale=sigma_1)
	d0 = (1-pie)*sp.stats.norm.pdf(beta,loc=0,scale=sigma_0)
	p = d1/(d0+d1)
	gamma = np.random.binomial(1,p)
	return(gamma)

def sample_pie(gamma,pie_a,pie_b):
	a_new = np.sum(gamma)+pie_a
	b_new = np.sum(1-gamma)+pie_b
	pie_new = np.random.beta(a_new,b_new)
	return(pie_new)

def sample_sigma_1(beta,gamma,a_sigma,b_sigma):
	a_new = 0.5*np.sum(gamma)+a_sigma
	b_new = 0.5*np.sum(np.multiply(np.square(beta),gamma))+b_sigma
	sigma_1_neg2 =np.random.gamma(a_new,1.0/b_new)
	sigma_1_new = math.sqrt(1/sigma_1_neg2)
	return(sigma_1_new)

def sample_sigma_e(y,H,beta,C,alpha,a_e,b_e):
	n = len(y)
	a_new = float(n)/2+a_e
	resid = y - np.matmul(H,beta) - np.matmul(C,alpha)
	b_new = np.sum(np.square(resid))/2+b_e
	sigma_e_neg2 =np.random.gamma(a_new,1.0/b_new)
	sigma_e_new = math.sqrt(1/sigma_e_neg2)
	return(sigma_e_new)

def sample_alpha(y,H,beta,C,alpha,sigma_e):

	r,c = C.shape
	H_beta = np.matmul(H,beta)

	if c == 1:
		new_variance = 1/(np.linalg.norm(C[:,0])**2*sigma_e**-2)
		new_mean = new_variance*np.dot((y-H_beta),C[:,0])*sigma_e**-2
		alpha = np.random.normal(new_mean,math.sqrt(new_variance))
	else:
		for i in range(c):
			new_variance = 1/(np.linalg.norm(C[:,i])**2*sigma_e**-2)
			delta_C = np.delete(C,i,1)
			delta_alpha = np.delete(alpha,i)
			new_mean = new_variance*np.dot(y-np.matmul(delta_C,delta_alpha)-H_beta,C[:,i])*sigma_e**-2
			alpha[i] = np.random.normal(new_mean,math.sqrt(new_variance))
	return(alpha)

def sample_beta(y,C,alpha,H,beta,gamma,sigma_0,sigma_1,sigma_e,blocks):

	C_alpha = np.matmul(C,alpha)

	for i in range(blocks.shape[0]):
		indexs = np.arange(blocks[i,0],blocks[i,1])
		block_beta = np.array(beta[indexs])
		block_H = np.array(H[:,indexs])
		H_beta_complement = np.subtract(np.matmul(H,beta),np.matmul(block_H,block_beta))
		H_beta_complement = np.subtract(np.matmul(H,beta),np.matmul(block_H,block_beta))
		for j in range(len(indexs)):
			k = blocks[i,0] + j
			new_variance = 1/(np.linalg.norm(block_H[:,j])**2*sigma_e**-2+(1-gamma[k])*sigma_0**-2+gamma[k]*sigma_1**-2)
			new_mean = new_variance*np.dot(y-C_alpha-H_beta_complement-np.matmul(block_H,block_beta)+block_H[:,j]*block_beta[j],block_H[:,j])*sigma_e**-2
			block_beta[j] = np.random.normal(new_mean,math.sqrt(new_variance))
		beta[indexs] = block_beta
	return(beta)

def sampling(y,C,HapDM,sig0_initiate,sig1_initiate,sige_initiate,pie_initiate,step_size,iters,prefix):


	LOG = open(prefix+".log","w")
	##specify hyper parameters
	pie_a = 1
	pie_b = 1
	a_sigma = 1
	b_sigma = 1
	a_e = 1
	b_e = 1

	sigma_0 = sig0_initiate
	sigma_1 = sig1_initiate
	sigma_e = sige_initiate
	pie = pie_initiate

	
	print("initiate:",sigma_1,sigma_e,pie,file = LOG)

	#initiate beta,gamma and H matrix
	C_r, C_c = C.shape

	H = np.array(HapDM,dtype=np.float32)
	H = preprocessing.scale(H,with_std=False)
	
	#for simulation only
	H_r,H_c = H.shape

	#block generation step 
	m = H_c % step_size
	if m == 0:
		blocks_row = int(H_c / step_size) 
	else:
		blocks_row = int(H_c / step_size) +1
	

	blocks = np.zeros((blocks_row,2),dtype = int)


	for i in range(blocks_row-1):
		blocks[i,0] = i * step_size
		blocks[i,1] = blocks[i,0] + step_size
	blocks[blocks_row-1,0] = blocks[blocks_row-2,1]
	blocks[blocks_row-1,1] = H_c 

	#initiate alpha, alpha_trace, beta_trace and gamma_trace

	it = 0
	burn_in_iter = 2000
	trace = np.empty((iters-2000,7))
	alpha_trace = np.empty((iters-2000,C_c))
	gamma_trace = np.empty((iters-2000,H_c))
	beta_trace = np.empty((iters-2000,H_c))
	top5_beta_trace = np.empty((iters-2000,5))

	alpha = np.random.random(size = C_c)
	gamma = np.random.binomial(1,pie,H_c)
	beta = np.array(np.zeros(H_c,dtype=np.float32))

	for i in range(H_c):
		if gamma[i] == 0:
			beta[i] = np.random.normal(0,sigma_0)
		else:
			beta[i] = np.random.normal(0,sigma_1) 

	#start sampling

	while it < iters:
		beta_pre = np.array(beta)
		gamma_pre = np.array(gamma)
		alpha_pre = np.array(alpha)
		sigma_1_pre = sigma_1
		sigma_e_pre = sigma_e
		pie_pre = pie

		sigma_1_update = sample_sigma_1(beta_pre,gamma_pre,a_sigma,b_sigma)
		if sigma_1_update < 0.05:
			sigma_1_update = 0.05
			pie_update = 0
		else:
			pie_update = sample_pie(gamma_pre,pie_a,pie_b)
		sigma_e_update = sample_sigma_e(y,H,beta_pre,C,alpha_pre,a_e,b_e)
		gamma_update = sample_gamma(beta_pre,sigma_0,sigma_1_update,pie_update)
		alpha_update = sample_alpha(y,H,beta_pre,C,alpha_pre,sigma_e_update)
		before = time.time()
		beta_update = sample_beta(y,C,alpha_update,H,beta_pre,gamma_update,sigma_0,sigma_1_update,sigma_e_update,blocks)
		after = time.time()
		genetic_var = np.var(np.matmul(H,beta_update))
		pheno_var = np.var(y - np.matmul(C,alpha_update))

		large_beta = np.absolute(beta_update) > 0.3
		#print(np.sort(np.absolute(beta_update))[::-1][:10])
		large_beta_ratio = np.sum(large_beta) / len(beta_update)
		large_beta_heritability = np.var(np.matmul(H[:,large_beta],beta_update[large_beta])) / pheno_var
		total_heritability = genetic_var / pheno_var

		if it > 100 and large_beta_heritability > 1 and large_beta_heritability > total_heritability:
			print("unrealistic beta sample",it,genetic_var,pheno_var,large_beta_heritability,total_heritability)
			continue

		# elif it > 100 and large_beta_heritability > total_heritability:
		# 	print("unrealistic beta sample",it,genetic_var,pheno_var,large_beta_heritability,total_heritability)
		# 	continue

		else:
			beta = np.array(beta_update)
			gamma = np.array(gamma_update)
			alpha = np.array(alpha_update)
			sigma_1 = sigma_1_update
			sigma_e = sigma_e_update
			pie = pie_update

			#print(it,str(after - before),large_beta_ratio,large_beta_heritability,total_heritability)

			if it >= burn_in_iter:
				trace[it-burn_in_iter,:] = [sigma_1,sigma_e,large_beta_ratio,large_beta_heritability,total_heritability,pie,it]
				gamma_trace[it-burn_in_iter,:] = gamma
				beta_trace[it-burn_in_iter,:] = beta
				alpha_trace[it-burn_in_iter,:] = alpha
				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

			if it >= burn_in_iter + 7999: # after burn-in iterations, test convergence

				max_z = []
				
				for a in range(C_c):
					after_burnin_alpha = alpha_trace[:,a]
					alpha_zscores = pm3.geweke(after_burnin_alpha)[:,1]
					max_z.append(np.amax(np.absolute(alpha_zscores)))

				for b in range(5):
					after_burnin_beta = top5_beta_trace[:,b]
					beta_zscores = pm3.geweke(after_burnin_beta)[:,1]
					max_z.append(np.amax(np.absolute(beta_zscores)))

				#convergence for pie
				after_burnin_pie = trace[:,2]
				pie_zscores = pm3.geweke(after_burnin_pie)[:,1]
				max_z.append(np.amax(np.absolute(pie_zscores)))

				#convergence for large_heritability
				after_burnin_var = trace[:,3]
				var_zscores = pm3.geweke(after_burnin_var)[:,1]
				max_z.append(np.amax(np.absolute(var_zscores)))

				#convergence for sigma_1
				# after_burnin_sigma1 = trace[:,0]
				# sigma1_zscores = pm3.geweke(after_burnin_sigma1)[:,1]
				# max_z.append(np.amax(np.absolute(sigma1_zscores)))

				#convergence for sigma_e
				after_burnin_sigmae = trace[:,1]
				sigmae_zscores = pm3.geweke(after_burnin_sigmae)[:,1]
				max_z.append(np.amax(np.absolute(sigmae_zscores)))
				
				if  np.amax(max_z) < 1.5:
					print("convergence has been reached at %i iterations." %(it),file=LOG)
					break

				else:
					trace_ = np.empty((1000,7))
					gamma_trace_ = np.empty((1000,H_c))
					beta_trace_ = np.empty((1000,H_c))
					alpha_trace_ = np.empty((1000,C_c))
					top5_beta_trace_ = np.empty((1000,5))

					trace = np.concatenate((trace[-(iters - burn_in_iter-1000):,:],trace_),axis=0)
					gamma_trace = np.concatenate((gamma_trace[-(iters - burn_in_iter-1000):,:],gamma_trace_),axis=0)
					beta_trace = np.concatenate((beta_trace[-(iters - burn_in_iter-1000):,:],beta_trace_),axis=0)
					alpha_trace = np.concatenate((alpha_trace[-(iters - burn_in_iter-1000):,:],alpha_trace_),axis=0)
					top5_beta_trace = np.concatenate((top5_beta_trace[-(iters - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

					burn_in_iter += 1000
					iters += 1000

			if (it - burn_in_iter) >= 0 and (it - burn_in_iter ) % 1000 == 0:
				print("%i iterations have sampled" %(it), str(after - before),trace[it-burn_in_iter,:],file=LOG)

			it += 1
	
	LOG.close()
	trace = pd.DataFrame(trace)
	alpha_trace = pd.DataFrame(alpha_trace)
	beta_trace = pd.DataFrame(beta_trace)
	gamma_trace = pd.DataFrame(gamma_trace)
	return(trace,alpha_trace,beta_trace,gamma_trace)

def sampling_w_annotation(y,C,HapDM,annotation,sig0_initiate,sig1_initiate,sige_initiate,pie_initiate,step_size,iters,prefix):

	##specify hyper parameters
	pie_a = 1
	pie_b = 1
	a_sigma = 1
	b_sigma = 1
	a_e = 1
	b_e = 1

	sigma_0 = sig0_initiate
	sigma_1 = sig1_initiate
	sigma_e = sige_initiate
	pie = pie_initiate
	
	print("initiate:",sigma_1,sigma_e,pie)

	#initiate beta,gamma and H matrix
	C_r, C_c = C.shape

	H = np.array(HapDM,dtype=np.float32)
	H = preprocessing.scale(H,with_std=False)
	H_r,H_c = H.shape


	#block generation step 
	m = H_c % step_size
	if m == 0:
		blocks_row = int(H_c / step_size) 
	else:
		blocks_row = int(H_c / step_size) +1
	

	blocks = np.zeros((blocks_row,2),dtype = int)


	for i in range(blocks_row-1):
		blocks[i,0] = i * step_size
		blocks[i,1] = blocks[i,0] + step_size
	blocks[blocks_row-1,0] = blocks[blocks_row-2,1]
	blocks[blocks_row-1,1] = H_c 

	#initiate alpha, alpha_trace, beta_trace and gamma_trace

	it = 0
	burn_in_iter = 2000
	trace = np.empty((iters-2000,6))
	alpha_trace = np.empty((iters-2000,C_c))
	theta_trace = np.empty((iters-2000,annotation.shape[1]))
	gamma_trace = np.empty((iters-2000,H_c))
	beta_trace = np.empty((iters-2000,H_c))
	top5_beta_trace = np.empty((iters-2000,5))


	alpha = np.random.random(size = C_c)

	theta = np.append(sp.stats.norm.ppf(pie_initiate),np.repeat(0,annotation.shape[1]-1))
	
	Z = np.matmul(annotation,theta)
	pie = sp.stats.norm.cdf(Z)
	print(pie)

	gamma = np.random.binomial(1,pie_initiate,H_c)
	
	beta = np.array(np.zeros(H_c,dtype=np.float32))
	for i in range(H_c):
		if gamma[i] == 0:
			beta[i] = np.random.normal(0,sigma_0)
		else:
			beta[i] = np.random.normal(0,sigma_1) 

	print(beta.dtype)

	
	#start sampling

	while it < iters:
		beta_pre = np.array(beta)
		gamma_pre = np.array(gamma)
		alpha_pre = np.array(alpha)
		sigma_1_pre = sigma_1
		sigma_e_pre = sigma_e
		theta_pre = np.array(theta)
		Z_pre = np.array(Z)
		before = time.time()

		Z_update = sample_Z(theta_pre,gamma_pre,annotation)
		theta_update = sample_theta(annotation,Z_update)
		gamma_update = sample_gamma_annotation(beta_pre,gamma_pre,sigma_0,sigma_1_pre,annotation,theta_update)
		sigma_1_update = sample_sigma_1(beta_pre,gamma_update,a_sigma,b_sigma)
		sigma_e_update = sample_sigma_e(y,H,beta_pre,C,alpha_pre,a_e,b_e)
		alpha_update = sample_alpha(y,H,beta_pre,C,alpha_pre,sigma_e_update)
		beta_update = sample_beta(y,C,alpha_update,H,beta_pre,gamma_update,sigma_0,sigma_1_update,sigma_e_update,blocks)
		after = time.time()
		genetic_var = np.var(np.matmul(H,beta_update))
		pheno_var = np.var(y - np.matmul(C,alpha_update))

		large_beta = np.absolute(beta_update) > 0.3
		large_beta_ratio = np.sum(large_beta) / len(beta_update)

		large_pie = sp.stats.norm.cdf(Z_update) > 0.1
		large_pie_ratio = np.sum(large_pie) / len(Z_update)
		large_beta_heritability = np.var(np.matmul(H[:,large_beta],beta[large_beta])) / pheno_var
		total_heritability = genetic_var / pheno_var


		if it > 100 and genetic_var > pheno_var and large_beta_heritability > 1 and large_beta_heritability > total_heritability:
			print("unrealistic beta sample",genetic_var,pheno_var)
			continue
		else:
			beta = np.array(beta_update)
			gamma = np.array(gamma_update)
			alpha = np.array(alpha_update)
			theta = np.array(theta_update)
			Z = np.array(Z_update)
			sigma_1 = sigma_1_update
			sigma_e = sigma_e_update
			
			print(it,str(after - before),large_beta_ratio,large_beta_heritability,total_heritability)

			if it >= burn_in_iter:
				trace[it-burn_in_iter,:] = [sigma_1,sigma_e,large_pie_ratio,large_beta_ratio,large_beta_heritability,total_heritability]
				gamma_trace[it-burn_in_iter,:] = gamma
				beta_trace[it-burn_in_iter,:] = beta
				alpha_trace[it-burn_in_iter,:] = alpha
				theta_trace[it-burn_in_iter,:] = theta
				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

			if it >= burn_in_iter + 7999: # after burn-in iterations, test convergence

				max_z = []

				# for t in range(len(theta)):
				#  	after_burnin_theta = theta_trace[:,t]
				#  	theta_zscores = pm3.geweke(after_burnin_theta)[:,1]
				#  	max_z.append(np.amax(np.absolute(theta_zscores)))

				for a in range(C_c):
					after_burnin_alpha = alpha_trace[:,a]
					alpha_zscores = pm3.geweke(after_burnin_alpha)[:,1]
					max_z.append(np.amax(np.absolute(alpha_zscores)))

				for b in range(5):
					after_burnin_beta = top5_beta_trace[:,b]
					beta_zscores = pm3.geweke(after_burnin_beta)[:,1]
					max_z.append(np.amax(np.absolute(beta_zscores)))

				#convergence for large pie ratio
				after_burnin_pie = trace[:,2]
				pie_zscores = pm3.geweke(after_burnin_pie)[:,1]
				max_z.append(np.amax(np.absolute(pie_zscores)))

				#convergence for large beta ratio
				after_burnin_beta_ratio = trace[:,3]
				pie_zscores = pm3.geweke(after_burnin_beta_ratio)[:,1]
				max_z.append(np.amax(np.absolute(pie_zscores)))

				#convergence for genetics variance
				after_burnin_var = trace[:,4]
				var_zscores = pm3.geweke(after_burnin_var)[:,1]
				max_z.append(np.amax(np.absolute(var_zscores)))

				#convergence for sigma_1
				# after_burnin_sigma1 = trace[:,0]
				# sigma1_zscores = pm3.geweke(after_burnin_sigma1)[:,1]
				# max_z.append(np.amax(np.absolute(sigma1_zscores)))

				#convergence for sigma_e
				after_burnin_sigmae = trace[:,1]
				sigmae_zscores = pm3.geweke(after_burnin_sigmae)[:,1]
				max_z.append(np.amax(np.absolute(sigmae_zscores)))
				
				if  np.amax(max_z) < 1.5:
					print("convergence has been reached at %i iterations." %(it))
					break

				else:
					trace_ = np.empty((1000,6))
					gamma_trace_ = np.empty((1000,H_c))
					beta_trace_ = np.empty((1000,H_c))
					alpha_trace_ = np.empty((1000,C_c))
					theta_trace_ = np.empty((1000,annotation.shape[1]))
					top5_beta_trace_ = np.empty((1000,5))

					trace = np.concatenate((trace[-(iters - burn_in_iter-1000):,:],trace_),axis=0)
					gamma_trace = np.concatenate((gamma_trace[-(iters - burn_in_iter-1000):,:],gamma_trace_),axis=0)
					beta_trace = np.concatenate((beta_trace[-(iters - burn_in_iter-1000):,:],beta_trace_),axis=0)
					alpha_trace = np.concatenate((alpha_trace[-(iters - burn_in_iter-1000):,:],alpha_trace_),axis=0)
					theta_trace = np.concatenate((theta_trace[-(iters - burn_in_iter-1000):,:],theta_trace_),axis=0)
					top5_beta_trace = np.concatenate((top5_beta_trace[-(iters - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

					burn_in_iter += 1000
					iters += 1000

			if (it - burn_in_iter) >= 0 and (it - burn_in_iter ) % 1000 == 0:
				print("%i iterations have sampled" %(it), str(after - before),trace[it-burn_in_iter,:])

			it += 1

	trace = pd.DataFrame(trace)
	alpha_trace = pd.DataFrame(alpha_trace)
	beta_trace = pd.DataFrame(beta_trace)
	gamma_trace = pd.DataFrame(gamma_trace)
	theta_trace = pd.DataFrame(theta_trace)
	return(trace,alpha_trace,beta_trace,gamma_trace,theta_trace)


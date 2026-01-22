
import numpy as np
import scipy as sp
import math
import pandas as pd 
import time
import geweke
import os
import gc
import utility_functions as uf
from numba import njit


def sample_gamma_annotation(beta,gamma,sigma_0,sigma_1,A,theta):
	inv_pie = np.matmul(A,theta)
	pie = sp.stats.norm.cdf(inv_pie)
	#beta is a list of marginal beta_i
	p = np.empty(len(beta))
	d1 = np.multiply(pie,sp.stats.norm.pdf(beta,loc=0,scale=sigma_1))
	d0 = np.multiply((1-pie),sp.stats.norm.pdf(beta,loc=0,scale=sigma_0))
	p = d1/(d0+d1)
	gamma = np.random.binomial(1,p).astype(np.int8)
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

def sample_sigma_e(y,H_beta,C_alpha,a_e,b_e):
	n = len(y)
	a_new = float(n)/2+a_e
	resid = y - H_beta - C_alpha
	b_new = np.sum(np.square(resid))/2+b_e
	sigma_e_neg2 =np.random.gamma(a_new,1.0/b_new)
	sigma_e_new = math.sqrt(1/sigma_e_neg2)
	return(sigma_e_new)

def sample_alpha(y,H_beta,C_alpha,C,alpha,sigma_e,C_norm_2):

	r,c = C.shape

	if c == 1:
		#new_variance = 1/(np.linalg.norm(C[:,0])**2*sigma_e**-2)
		new_variance = 1/(C_norm_2[0]*sigma_e**-2)
		new_mean = new_variance*np.dot((y-H_beta),C[:,0])*sigma_e**-2
		alpha = np.random.normal(new_mean,math.sqrt(new_variance))
		C_alpha = C[:,0] * alpha
	else:
		for i in range(c):
			#new_variance = 1/(np.linalg.norm(C[:,i])**2*sigma_e**-2)
			new_variance = 1/(C_norm_2[i]*sigma_e**-2)
			C_alpha_negi = C_alpha - C[:,i] * alpha[i]
			new_mean = new_variance*np.dot(y-C_alpha_negi-H_beta,C[:,i])*sigma_e**-2
			alpha[i] = np.random.normal(new_mean,math.sqrt(new_variance))
			C_alpha = C_alpha_negi + C[:,i] * alpha[i]

	return(alpha,C_alpha)

def sample_beta(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e,H_norm_2):

	sigma_e_neg2 = sigma_e**-2
	sigma_0_neg2 = sigma_0**-2
	sigma_1_neg2 = sigma_1**-2

	for i in range(len(beta)):
		H_beta_negi = H_beta - H[:,i] * beta[i]
		residual = y - C_alpha -  H_beta + H[:,i] * beta[i]
		#new_variance = 1/(np.sum(H[:,i]**2)*sigma_e_neg2+(1-gamma[i])*sigma_0_neg2+gamma[i]*sigma_1_neg2)
		new_variance = 1/(H_norm_2[i]*sigma_e_neg2+(1-gamma[i])*sigma_0_neg2+gamma[i]*sigma_1_neg2)
		new_mean = new_variance*np.dot(residual,H[:,i])*sigma_e_neg2
		beta[i] = np.random.normal(new_mean,math.sqrt(new_variance))
		H_beta = H_beta_negi + H[:,i] * beta[i]

	# for i in range(blocks.shape[0]):
	# 	indexs = np.arange(blocks[i,0],blocks[i,1])
	# 	block_beta = np.array(beta[indexs])
	# 	block_H = np.array(H[:,indexs])
	# 	H_beta_complement = np.subtract(np.matmul(H,beta),np.matmul(block_H,block_beta))
	# 	H_beta_complement = np.subtract(np.matmul(H,beta),np.matmul(block_H,block_beta))
	# 	for j in range(len(indexs)):
	# 		k = blocks[i,0] + j
	# 		new_variance = 1/(np.linalg.norm(block_H[:,j])**2*sigma_e**-2+(1-gamma[k])*sigma_0**-2+gamma[k]*sigma_1**-2)
	# 		new_mean = new_variance*np.dot(y-C_alpha-H_beta_complement-np.matmul(block_H,block_beta)+block_H[:,j]*block_beta[j],block_H[:,j])*sigma_e**-2
	# 		block_beta[j] = np.random.normal(new_mean,math.sqrt(new_variance))
	# 	beta[indexs] = block_beta
	return(beta,H_beta)

@njit
def sample_beta_numba(y, C_alpha, H_beta, H, beta, gamma, sigma_0, sigma_1, sigma_e, H_norm_2):
	sigma_e_neg2 = sigma_e ** -2
	sigma_0_neg2 = sigma_0 ** -2
	sigma_1_neg2 = sigma_1 ** -2
	ncols = beta.shape[0]
	nrows = y.shape[0]
    
	for i in range(ncols):

		for r in range(nrows):
			H_beta[r] -= H[r, i] * beta[i]

        # Compute the dot product over the column using the updated H_beta.
		dot_val = 0.0
		for r in range(nrows):
            # residual = y[r] - C_alpha[r] - H_beta[r]
			res_val = y[r] - C_alpha[r] - H_beta[r] 
			dot_val += res_val * H[r, i]
        
		new_variance = 1.0 / (H_norm_2[i]*sigma_e_neg2 + (1 - gamma[i])*sigma_0_neg2 + gamma[i]*sigma_1_neg2)
		new_mean = new_variance * sigma_e_neg2 * dot_val
        
        # Sample new beta using standard normal (Numba supports np.random.randn)
		beta[i] = new_mean + math.sqrt(new_variance) * np.random.randn()
       
        # Update H_beta with the new contribution.
		for r in range(nrows):
			H_beta[r] += H[r, i] * beta[i]
    
	return (beta, H_beta)

def sampling(verbose,y,C,H,sig0_initiate,iters,prefix,block_haplotypes,block_positions,num,trace_container,gamma_container,beta_container,alpha_container,convergence_container):

	## set random seed for the process
	np.random.seed(int(time.time()) + os.getpid())

	#initiate beta,gamma and H matrix
	# H = np.array(HapDM)

	H_r,H_c = H.shape


	C_c = C.shape[1]


	##specify hyper parameters
	pie_a = 1
	pie_b = H_c / 50
	a_sigma = 1
	b_sigma = 1
	a_e = 1
	b_e = 1

	H_var = np.sum(np.var(H,axis=0))
	sigma_0 = np.sqrt(np.var(y) / H_var * sig0_initiate)
	sigma_1 = math.sqrt(1/np.random.gamma(a_sigma,b_sigma))
	sigma_e = math.sqrt(1/np.random.gamma(a_e,b_e))
	pie = np.random.beta(pie_a,pie_b)
	
	if verbose > 0:
		print("To set the background variation %f of the total phenotypic variation. We set the sigma 0 to be %f" %(sig0_initiate,sigma_0) )
		print("initiation for chain %i:" %(num) ,sigma_1,sigma_e,pie)

	#print("initiation for chain %i:" %(num) ,sigma_1,sigma_e,pie)

	
	#initiate alpha, alpha_trace, beta_trace and gamma_trace

	it = 0
	burn_in_iter = 2000
	step_size =2000
	
	convergence_start_iter = burn_in_iter
	convergence_end_iter = np.array(range(convergence_start_iter*2,convergence_start_iter+step_size*4,step_size))
	convergence_iter = convergence_start_iter+step_size*3

	trace = np.empty((convergence_end_iter[-1]-burn_in_iter,7))
	top5_beta_trace = np.empty((convergence_end_iter[-1]-burn_in_iter,5))


	alpha = np.random.random(size = C_c)
	gamma = np.random.binomial(1,pie,H_c).astype(np.int8)
	beta = np.array(np.zeros(H_c))

	for i in range(H_c):
		if gamma[i] == 0:
			beta[i] = np.random.normal(0,sigma_0)
		else:
			beta[i] = np.random.normal(0,sigma_1) 

	#start sampling

	H_beta = np.matmul(H,beta)
	C_alpha = np.matmul(C,alpha)


	## precompute some variables 

	C_norm_2 = np.sum(C**2,axis=0)
	H_norm_2 = np.sum(H**2,axis=0)


	while it < convergence_iter:
		before = time.time()

		sigma_1 = sample_sigma_1(beta,gamma,a_sigma,b_sigma)
		if sigma_1 < sigma_0 * 5:
			sigma_1 = sigma_0 * 5
			pie = 0
		else:
			pie = sample_pie(gamma,pie_a,pie_b)
		sigma_e = sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
		gamma = sample_gamma(beta,sigma_0,sigma_1,pie)
		alpha,C_alpha = sample_alpha(y,H_beta,C_alpha,C,alpha,sigma_e,C_norm_2)
		#beta,H_beta = sample_beta(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e,H_norm_2)
		beta,H_beta = sample_beta_numba(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e,H_norm_2)
		genetic_var = np.var(H_beta)
		pheno_var = np.var(y - C_alpha)
		large_beta = np.absolute(beta) > 0.3
		large_beta_ratio = np.sum(large_beta) / len(beta)
		total_heritability = genetic_var / pheno_var

		if C_c == 1:
			alpha_norm = alpha
		else:
			alpha_norm = np.linalg.norm(alpha, ord=2)
		beta_norm = np.linalg.norm(beta, ord=2)

		after = time.time()
		if (it > 2000 and total_heritability > 1):
			if verbose > 0:
				print("unrealistic beta sample",it,genetic_var,pheno_var,total_heritability)
			continue

		else:
			if verbose > 1:
				print(num,it,str(after - before),sigma_1,sigma_e,large_beta_ratio,total_heritability,sum(gamma))

			if it >= burn_in_iter:
				trace[it-burn_in_iter,:] = [alpha_norm,beta_norm,sigma_1,sigma_e,large_beta_ratio,total_heritability,sum(gamma)]
				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

			if it == convergence_end_iter[-1] - 1:
				
				num_convergence_test = len(convergence_end_iter)

				convergence_scores = np.zeros(len(convergence_end_iter))

				for s in range(num_convergence_test):
					convergence_scores[s] = uf.convergence_geweke_test(trace,top5_beta_trace,convergence_start_iter-burn_in_iter,convergence_end_iter[s]-burn_in_iter)

				if np.sum(convergence_scores) == num_convergence_test:
					convergence_container[num] = 1

					if verbose > 0:
						print("convergence has been reached at %i iterations for chain %i. The MCMC Chain has entered a stationary stage" %(it,num))
						print("trace values:", trace[it-burn_in_iter,:])
					break
				else:
					trace_ = np.empty((1000,7))
					top5_beta_trace_ = np.empty((1000,5))


					trace = np.concatenate((trace[-(convergence_iter - burn_in_iter-1000):,:],trace_),axis=0)
					top5_beta_trace = np.concatenate((top5_beta_trace[-(convergence_iter - burn_in_iter-1000):,:],top5_beta_trace_),axis = 0)

					burn_in_iter += 1000
					convergence_iter += 1000

					convergence_start_iter += 1000
					convergence_end_iter += 1000

					#print(it,burn_in_iter,convergence_iter,convergence_start_iter,convergence_end_iter,trace.shape)

			it += 1

			if it > 100000: 
				convergence_container[num] = 0
				break
	
	if convergence_container[num] == 1:

		## MCMC draws for posterior mean

		posterior_draws = 10000

		alpha_mean = np.zeros(C_c)
		beta_mean = np.zeros(H_c)
		gamma_sum = np.zeros(H_c)

		alpha_M2 = np.zeros(C_c)
		beta_M2 = np.zeros(H_c)

		posterior_trace = np.empty((posterior_draws,7))

		alpha_trace = np.empty((posterior_draws,C_c))

		it = 0

		while it < posterior_draws:
		
			before = time.time()
			sigma_1 = sample_sigma_1(beta,gamma,a_sigma,b_sigma)
			if sigma_1 < sigma_0 * 5:
				sigma_1 = sigma_0 * 5
				pie = 0
			else:
				pie = sample_pie(gamma,pie_a,pie_b)
			sigma_e = sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
			gamma = sample_gamma(beta,sigma_0,sigma_1,pie)
			alpha,C_alpha = sample_alpha(y,H_beta,C_alpha,C,alpha,sigma_e,C_norm_2)
			beta,H_beta = sample_beta_numba(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e,H_norm_2)
			genetic_var = np.var(H_beta)
			pheno_var = np.var(y - C_alpha)
			large_beta = np.absolute(beta) > 0.3
			large_beta_ratio = np.sum(large_beta) / len(beta)
			total_heritability = genetic_var / pheno_var
			if C_c == 1:
				alpha_norm = alpha
			else:
				alpha_norm = np.linalg.norm(alpha, ord=2)
			beta_norm = np.linalg.norm(beta, ord=2)
			after = time.time()
			if total_heritability > 1:
				if verbose > 0:
					print("unrealistic beta sample",it,genetic_var,pheno_var,total_heritability)
				continue

			else:
				if verbose >1 :
					print(it,str(after - before),pie,sigma_1,sigma_e,sum(gamma),large_beta_ratio,max(abs(beta)),total_heritability)

				posterior_trace[it,:] = [alpha_norm,beta_norm,sigma_1,sigma_e,large_beta_ratio,total_heritability,sum(gamma)]
				alpha_trace[it,:] = alpha
				beta_mean,beta_M2 = uf.welford(beta_mean,beta_M2,beta,it)
				alpha_mean,alpha_M2 = uf.welford(alpha_mean,alpha_M2,alpha,it)
				gamma_sum += gamma

				if verbose > 0:
					if it > 0 and it % 2000 == 0:
						print("Posterior draws: %i iterations have been sampled for chain %i" %(it,num), str(after - before),posterior_trace[it,:])
				it += 1

		trace_container[num] = posterior_trace

		#alpha values
		alpha_container[num] = {'avg': alpha_mean,
								'M2': alpha_M2}

		#beta values
		beta_container[num] = {'avg':beta_mean,
								'M2':beta_M2}

		gamma_container[num] = gamma_sum / posterior_draws

	else:
		trace_container[num] = []

		#alpha values
		alpha_container[num] = {'avg': [],
								'M2': []}

		#beta values
		beta_container[num] = {'avg':[],
								'M2':[]}

		gamma_container[num] = []


def sampling_w_annotation(y,C,HapDM,annotation,sig0_initiate,sig1_initiate,sige_initiate,pie_initiate,step_size,iters,prefix):

	#initiate beta,gamma and H matrix
	H_r,H_c = H.shape

	##specify hyper parameters
	pie_a = 1
	pie_b = H_c / 10
	a_sigma = 1
	b_sigma = 1
	a_e = 1
	b_e = 1

	sigma_0 = sig0_initiate
	sigma_1 = sig1_initiate
	sigma_e = sige_initiate
	pie = pie_initiate
	
	print("initiate:",sigma_1,sigma_e,pie)

	H = np.array(HapDM)

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

	gamma = np.random.binomial(1,pie_initiate,H_c)
	
	beta = np.array(np.zeros(H_c))
	for i in range(H_c):
		if gamma[i] == 0:
			beta[i] = np.random.normal(0,sigma_0)
		else:
			beta[i] = np.random.normal(0,sigma_1) 


	H_beta = np.matmul(H,beta)
	C_alpha = np.matmul(C,alpha)
	#start sampling

	while it < iters:
		before = time.time()
		Z = sample_Z(theta,gamma,annotation)
		theta = sample_theta(annotation,Z)
		gamma = sample_gamma_annotation(beta,gamma,sigma_0,sigma_1,annotation,theta)
		sigma_1 = sample_sigma_1(beta,gamma,a_sigma,b_sigma)
		sigma_e = sample_sigma_e(y,H_beta,C_alpha,a_e,b_e)
		alpha,C_alpha = sample_alpha(y,H_beta,C_alpha,C,alpha,sigma_e)
		beta,H_beta = sample_beta(y,C_alpha,H_beta,H,beta,gamma,sigma_0,sigma_1,sigma_e)
		after = time.time()
		genetic_var = np.var(H_beta)
		
		pheno_var = np.var(y - C_alpha)
		large_beta = np.absolute(beta) > 0.3
		large_beta_ratio = np.sum(large_beta) / len(beta)
		large_pie = sp.stats.norm.cdf(Z_update) > 0.1
		large_pie_ratio = np.sum(large_pie) / len(Z_update)
		total_heritability = genetic_var / pheno_var


		if it > 100 and  total_heritability > 1:
			print("unrealistic beta sample",genetic_var,pheno_var)
			continue
		else:
			if it >= burn_in_iter:
				trace[it-burn_in_iter,:] = [it,sigma_1,sigma_e,large_beta_ratio,large_pie_ratio,total_heritability]
				gamma_trace[it-burn_in_iter,:] = gamma
				beta_trace[it-burn_in_iter,:] = beta
				alpha_trace[it-burn_in_iter,:] = alpha
				theta_trace[it-burn_in_iter,:] = theta
				top5_beta_trace[it-burn_in_iter,:] = np.sort(np.absolute(beta))[::-1][:5]

			if it >= burn_in_iter + 9999: # after burn-in iterations, test convergence

				max_z = []

				# for t in range(len(theta)):
				#  	after_burnin_theta = theta_trace[:,t]
				#  	theta_zscores = pm3.geweke(after_burnin_theta)[:,1]
				#  	max_z.append(np.amax(np.absolute(theta_zscores)))

				for a in range(C_c):
					after_burnin_alpha = alpha_trace[:,a]
					alpha_zscores = geweke.geweke(after_burnin_alpha)[:,1]
					max_z.append(np.amax(np.absolute(alpha_zscores)))

				for b in range(5):
					after_burnin_beta = top5_beta_trace[:,b]
					beta_zscores = geweke.geweke(after_burnin_beta)[:,1]
					max_z.append(np.amax(np.absolute(beta_zscores)))

				#convergence for large beta ratio
				after_burnin_pie = trace[:,4]
				pie_zscores = geweke.geweke(after_burnin_pie)[:,1]
				max_z.append(np.amax(np.absolute(pie_zscores)))

				#convergence for large pi ratio
				after_burnin_beta_ratio = trace[:,3]
				pie_zscores = geweke.geweke(after_burnin_beta_ratio)[:,1]
				max_z.append(np.amax(np.absolute(pie_zscores)))

				#convergence for total heritability
				after_burnin_var = trace[:,5]
				var_zscores = geweke.geweke(after_burnin_var)[:,1]
				max_z.append(np.amax(np.absolute(var_zscores)))

				#convergence for sigma_1
				after_burnin_sigma1 = trace[:,1]
				sigma1_zscores = geweke.geweke(after_burnin_sigma1)[:,1]
				max_z.append(np.amax(np.absolute(sigma1_zscores)))

				#convergence for sigma_e
				after_burnin_sigmae = trace[:,2]
				sigmae_zscores = geweke.geweke(after_burnin_sigmae)[:,1]
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



import re
import numpy as np
import pandas as pd
import scipy
import networkx as nx
import argparse
import math

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import AffinityPropagation
from pyclustering.cluster.gmeans import gmeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

import geweke
from cyvcf2 import VCF



## TO WORK AROUND THE NUMPY WARNING BUG

import warnings
np.warnings = warnings


def parse_arguments_mapping():

	parser = argparse.ArgumentParser()
	parser.add_argument('-i',type = str, action = 'store', dest ='input', help = 'the input haplotypeDM file')
	parser.add_argument('-c',type = str, action = 'store', dest ='covariates',help = 'the covariate file, tab deliminated without header')
	parser.add_argument('-y',type = str, action = 'store', dest ='phenotype',help = 'the phenotype file with one column and no header')
	parser.add_argument('-a',type = str, action = 'store', dest = 'annotation',help = 'the annotation file (beta version)')
	parser.add_argument('-n',type = int, action = 'store', default = 5, dest = "num", help = 'number of MCMC chains run parallelly. Default: 5')
	parser.add_argument('-m',type = int, action = 'store', default = 1, dest = 'mode',help = "1:no annotation; 2:with annotation file. Default: 1")
	parser.add_argument('-s0',type = float, action = 'store', dest = 's0',default = 0.05, help = "the proportion of phenotypic variation explained by background variables. Default: 0.05")
	parser.add_argument('-v',type = int, action = 'store', default = 0, dest = 'verbose', help = "verbose levels 0: no stdout; 1: convergence and minimal stdout; 2: per MCMC iteration stdout. Default: 0")
	parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of the output files")

	args = parser.parse_args()

	return(args)

def parse_arguments_haplotype():

	parser = argparse.ArgumentParser()
	parser.add_argument('-v',type = str, action= 'store',dest='vcf',help='the input vcf file. Required')
	parser.add_argument('-maf',type = float, action = 'store', dest = 'maf',default=0.05, help = "SNPs with minor allele frequency (MAF) ≥ this value are used for independent and bigLD block partitioning. Default: 0.05")
	parser.add_argument('-b',type = str, action= 'store',dest='block',default = "bigld",help="block partition method: bigld or previously defined blocks. Required")
	parser.add_argument('-r1',type = float, action = 'store', dest = 'corr',default=0.1, help = "r2 for independent LD blocks: 0-1. Default: 0.1")
	parser.add_argument('-w',type = int, action = 'store', dest = 'window',default = 100, help = "sliding window size for calcualting the independent LD blocks. Default: 100")
	parser.add_argument('-r2',type = float, action = 'store', dest = 'CLQcut',default=0.5, help = "r2 for bigLD fine LD block partitioning: 0-1. Default: 0.5)")
	parser.add_argument('-hc',type = str, action = 'store', dest = 'clustering',help = "haplotype clustering method: xmeans, KNN, modularity, local. Default: xmeans",default = "xmeans")
	parser.add_argument('-o',type = str, action = 'store', dest = 'output',help = "the prefix of the output files")
	args = parser.parse_args()

	return(args)

def vcf_processing(vcf):
	hap_matrix_d1 = {} #haplotype 1 of individuals, key as chromosome number
	hap_matrix_d2 = {} #haplotype 2 of individuals, key as chromosome number
	variant_names = {}
	variant_positions = {} #key as chromosome number
	chromosome = [] #key as chromosome number and value as number of SNPs per chromosome

	for v in VCF(vcf):
		ch = v.CHROM
		if ch not in chromosome:
			chromosome.append(ch)
			variant_names[ch] = []
			hap_matrix_d1[ch] = []
			hap_matrix_d2[ch] = []
			variant_positions[ch] = []

		if v.ID == None:
			variant_names[ch].append(str(ch) + "_" + str(v.POS))
		else:
			variant_names[ch].append(v.ID)		
		variant_positions[ch].append(v.POS)
		hap_matrix_d1[ch].append(np.array(v.genotypes, dtype=np.int16)[:,0])
		hap_matrix_d2[ch].append(np.array(v.genotypes, dtype=np.int16)[:,1])

		phased = np.sum(np.array(v.genotypes, dtype=np.int16)[:,2])

		if phased != len(np.array(v.genotypes, dtype=np.int16)[:,0]):
			sys.exit("FOUND unphased genotype, please make sure used completely phased vcf")

	for ch in chromosome:
		hap_matrix_d1[ch] = np.vstack(hap_matrix_d1[ch])
		hap_matrix_d2[ch] = np.vstack(hap_matrix_d2[ch])

	return(hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions,chromosome)

def convert_independent_genomewide_breakpoints(common_breakpoints,common_index,r):
	gw_breakpoints = []
	for i in range(len(common_breakpoints)):  ## there is a small bug issue when the first or the last snp is the block itself
		if len(common_breakpoints[i]) == 1:
			if common_breakpoints[i][0] == 0:
				left = 0
				right = common_index[common_breakpoints[i+1][0]] -1 
			else:
				left = common_index[common_breakpoints[i][0] -1] + 1
				right = common_index[common_breakpoints[i][0]]
		else:
			common_left = common_breakpoints[i][0]
			common_right = common_breakpoints[i][1]
			#print(common_left,common_right)
			if common_left == 0:
				left = 0
				right = common_index[common_right]
			elif common_right == len(common_index) -1:
				left = common_index[common_left - 1] + 1
				right = r -1
			else:
				left = common_index[common_left - 1] + 1
				right = common_index[common_right]
		gw_breakpoints.append([left,right])

	j = 0
	while j < len(gw_breakpoints):
		if gw_breakpoints[j][1] - gw_breakpoints[j][0] < 10:
			if j == 0:
				gw_breakpoints[j+1] = [gw_breakpoints[j][0],gw_breakpoints[j+1][1]]
				del gw_breakpoints[j]
			else:
				gw_breakpoints[j-1] = [gw_breakpoints[j-1][0],gw_breakpoints[j][1]]
				del gw_breakpoints[j]
		else:
			j += 1
	return(gw_breakpoints)


def convert_fine_genomewide_breakpoints(common_breakpoints,common_index,r,gw_independent_breakpoints):

	IndepLD_breakpoints = []

	for k in range(len(gw_independent_breakpoints)):
		IndepLD_breakpoints.append(gw_independent_breakpoints[k][0])
		IndepLD_breakpoints.append(gw_independent_breakpoints[k][1])

	gw_breakpoints = []
	for i in range(len(common_breakpoints)): #SHOULD NOT HAPPEN BUT IF HAPPENED THERE COULD BE A SMALL BUG
		common_left = common_breakpoints[i][0]
		common_right = common_breakpoints[i][1]
		if i == 0:
			left = 0
			right = common_index[common_right]

		elif i == len(common_breakpoints) -1:
			common_left_prev = common_breakpoints[i-1][1]
			left = common_index[common_left_prev] + 1
			right = r -1

		else:
			common_left_prev = common_breakpoints[i-1][1]
			left = common_index[common_left_prev] + 1
			right = common_index[common_right]
		gw_breakpoints.append([left,right])

	j = 0
	while j < len(gw_breakpoints):
		if gw_breakpoints[j][1] - gw_breakpoints[j][0] < 2:

			if gw_breakpoints[j][0] in IndepLD_breakpoints or gw_breakpoints[j][1] in IndepLD_breakpoints:
				if j == 0:
					gw_breakpoints[j+1] = [gw_breakpoints[j][0],gw_breakpoints[j+1][1]]
					del gw_breakpoints[j]
				else:
					j += 1
			else:
				gw_breakpoints[j-1] = [gw_breakpoints[j-1][0],gw_breakpoints[j][1]]
				del gw_breakpoints[j]
		else:
			j += 1
	return(gw_breakpoints)

def Standardize(X):
	r,c = X.shape
	mean = np.mean(X, axis=0)
	std = np.std(X,axis = 0)
	X_sd = (X - mean)/std
	return(X_sd)

def remove_duplicates(duplicated_list): 
	final_list = [] 
	for num in duplicated_list: 
		if num not in final_list: 
			final_list.append(int(num))  # change the data type into integers 
	return(final_list) 

def sortFirst(val): 
	return val[0] # sort by the first element

def find_blocks(joints_array,master_array):
	for i in range(len(joints_array)-1):
		index1 = master_array.index(joints_array[i])
		index2 = master_array.index(joints_array[i+1])
		del master_array[index1:index2]
	return(master_array)

def find_joints(joints_array):
	joints_dedup = remove_duplicates(joints_array)
	joints_dedup.sort()
	return(joints_dedup)

def find_index(joints_array,master_array):
	index = []
	for i in joints_array:
		index.append(master_array.index(i))
	return(index)

def haplotype_matrix(block_index,num_hap,hap):
	columns = [str(block_index)+'_'+str(i) for i in range(num_hap)]
	n,m = hap.shape
	haplotype_matrix = np.zeros((n,num_hap),dtype=int)
	unique_haplotype = []
	for i in range(n):
		haplotype = "".join(map(str,hap.values[i,:]))
		if haplotype not in unique_haplotype:
			unique_haplotype.append(haplotype)
		j = unique_haplotype.index(haplotype)
		haplotype_matrix[i,j] += 1
	return(haplotype_matrix)


def xmeans_clustering(array):
	# Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
	# start analysis.
	amount_initial_centers = 2
	initial_centers = kmeans_plusplus_initializer(array, amount_initial_centers).initialize()
	# Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
	# number of clusters that can be allocated is 30.
	xmeans_instance = xmeans(array, initial_centers, 30)
	xmeans_instance.process()
	# Extract clustering results: clusters and their centers
	clusters_ = xmeans_instance.get_clusters()
	clusters = [0]*len(array)
	for i in range(len(clusters_)):
		for j in clusters_[i]:
			clusters[j] = i
	print(clusters)
	return(clusters)

def gmeans_clustering(array):
	gmeans_instance = gmeans(array, repeat=10).process()
	clusters_ = gmeans_instance.get_clusters()

	clusters = [0]*len(array)
	for i in range(len(clusters_)):
		for j in clusters_[i]:
			clusters[j] = i
	return(clusters)

def affinity_propagation(np_array):
	AP = AffinityPropagation(random_state=None).fit(np_array)
	labels = AP.labels_
	return(labels)

def affinity_propagation_update(X, damping=0.7, preference=None):
	"""
    X: (n, m) haplotype matrix (0/1 or integer encoded)
    """
	X = np.asarray(X)
	n = X.shape[0]
	if n <= 1:
		return np.zeros(n, dtype=int)

    # Pairwise distance (Hamming or Euclidean)
	dists = squareform(pdist(X, metric="hamming"))

    # Convert distance to similarity
	S = - (dists ** 2)

    # Choose preference
	if preference is None:
		preference = np.median(S)   # very important!

	AP = AffinityPropagation(
		affinity="precomputed",
		damping=damping,
		preference=preference,
		random_state=0
	)

	labels = AP.fit_predict(S)
	return(labels)


def DBSCAN_clustering(np_array):
	r,c = np_array.shape
	clusters = DBSCAN(eps=0.1,min_samples=2,p=2).fit(np_array)
	labels = clusters.labels_
	return(labels)



def local_scale_Spectral(np_array,eps=1e-12,k_max_eigs=30):

	r,c =np_array.shape
	k = max(10, int(2*math.log(r)))
	k = min(k, r - 2)

	D = squareform(pdist(np_array,metric="hamming"))

	# sigma_i = distance to k-th neighbor (excluding self at position 0)
	D_sorted = np.sort(D, axis=1)
	sigma = D_sorted[:, k + 1]  # +1 because D_sorted[:,0] is self-distance 0
	sigma = np.maximum(sigma, eps)

	# Local scaling: exp( -d^2 / (sigma_i * sigma_j) )
	S = np.outer(sigma, sigma)
	S = np.maximum(S, eps)

	W = np.exp(-(D * D) / S)
	np.fill_diagonal(W, 0.0)
	L = csgraph.laplacian(W, normed=True)
	evals, evecs = np.linalg.eigh(L)

	# Use eigengap on the smallest part of the spectrum
	m = min(k_max_eigs, r - 1)
	small = evals[:m]
	gaps = np.diff(small)

	# Ignore the trivial first eigenvalue near 0; search gaps starting from index 1
	if len(gaps) < 2:
		n_clusters = 1
	else:
		j = 1 + np.argmax(gaps[1:])  # j is the split point
		n_clusters = j + 1

	# Spectral embedding: take first n_clusters eigenvectors
	V = evecs[:, :n_clusters]

	# Row-normalize (common in Ng-Jordan-Weiss style)
	V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)

	# Cluster embedding (Ward on Euclidean in embedding space is fine)
	Z = linkage(V_norm, method="ward")
	labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

	#print(max(labels))


	return(labels)

def local_scale_modularity(np_array,eps=1e-12):
	
	r,c =np_array.shape
	k = max(10, int(2*math.log(r)))
	k_knn = min(k, r - 1)
	k_sigma = min(k, r - 2)


	D = squareform(pdist(np_array,metric="hamming"))

	# sigma_i = distance to k-th neighbor (excluding self at position 0)
	D_sorted = np.sort(D, axis=1)
	sigma = D_sorted[:, k_sigma + 1]  # +1 because D_sorted[:,0] is self-distance 0
	sigma = np.maximum(sigma, eps)

	# Local scaling: exp( -d^2 / (sigma_i * sigma_j) )
	S = np.outer(sigma, sigma)
	S = np.maximum(S, eps)

	W = np.exp(-(D * D) / S)
	np.fill_diagonal(W, 0.0)


	connectivity = kneighbors_graph(X=np_array, n_neighbors=k_knn, mode='connectivity',metric="hamming",include_self=False)
	connectivity = 0.5 * (connectivity + connectivity.T) ## make connectivity symmetric

	connectivity_matrix = connectivity.toarray()
	weighted_connectivity = np.multiply(W,connectivity_matrix)
	G = nx.from_numpy_array(weighted_connectivity)
	## this is a more stable function that the results can be reproduced due to the seed setting
	louvain_partition = nx.community.louvain_communities(G,seed=0)
	labels = np.empty(r, dtype=int)
	for cid, nodes in enumerate(louvain_partition):
		labels[list(nodes)] = cid

	return(labels)

def KNN_Spectral(np_array,eps=1e-12, k_max_eigs=30):

	r,c =np_array.shape
	k = max(10, int(2*math.log(r)))
	k = min(k, r - 1)

	connectivity = kneighbors_graph(X=np_array, n_neighbors=k, mode='connectivity',metric="hamming",include_self=False)

	A = (1/2)*(connectivity + connectivity.T)
	
	L = csgraph.laplacian(A,normed = True)

	L = L.toarray()

	evals, evecs = np.linalg.eigh(L)

	# Use eigengap on the smallest part of the spectrum
	m = min(k_max_eigs, r - 1)
	small = evals[:m]
	gaps = np.diff(small)

	# Ignore the trivial first eigenvalue near 0; search gaps starting from index 1
	if len(gaps) < 2:
		n_clusters = 1
	else:
		j = 1 + np.argmax(gaps[1:])  # j is the split point
		n_clusters = j + 1

	# Spectral embedding: take first n_clusters eigenvectors
	V = evecs[:, :n_clusters]

	# Row-normalize (common in Ng-Jordan-Weiss style)
	V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)

	# Cluster embedding (Ward on Euclidean in embedding space is fine)
	Z = linkage(V_norm, method="ward")
	labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

	#print(max(labels))

	return(labels)

def BlockDM_generation(ch,r,hap_matrix_d1,hap_matrix_d2,geno_matrix,variant_names,variant_positions,fine_breakpoints,HaploBlock_matrix,haplotype_block_name,haplotype_marker_name,clutering_algorithm):
	hap_matrix_d1_pd = pd.DataFrame(np.transpose(hap_matrix_d1[ch]),columns=variant_names[ch])
	hap_matrix_d2_pd = pd.DataFrame(np.transpose(hap_matrix_d2[ch]),columns=variant_names[ch])

	HaploBlock_matrix_container = {}
	haplotype_block_name_container =[]
	haplotype_marker_name_container = []

	l = len(fine_breakpoints[ch])
	columns = []
	BLOCK_NAMES = []
	block_Dmatrix = pd.DataFrame(index=range(r),columns=columns)

	for i in range(l):
		fine_index1 = fine_breakpoints[ch][i][0]
		fine_index2 = fine_breakpoints[ch][i][1]
		hap1 = hap_matrix_d1_pd[variant_names[ch][fine_index1:fine_index2+1]]
		hap2 = hap_matrix_d2_pd[variant_names[ch][fine_index1:fine_index2+1]]
		haplotypes = pd.concat([hap1,hap2],ignore_index=True)
		haplotype_DM_,block_name,haplotype_names,marker_name = haplotype_DM_generator(block_index=i,
			clutering_algorithm =clutering_algorithm,
			haplotypes = haplotypes,
			n_clusters = 7,
			breakpoints = fine_breakpoints[ch],
			positions = variant_positions[ch],
			ch = ch)

		block_Dmatrix = pd.concat([block_Dmatrix,haplotype_DM_],axis=1,ignore_index=True)
		# BLOCK_NAMES.append(block_name)
		# haplotype_block_name_container.append(block_name)
		BLOCK_NAMES.extend(haplotype_names)
		haplotype_block_name_container.append(block_name)
		haplotype_marker_name_container.append(haplotype_names)
	block_Dmatrix.columns = BLOCK_NAMES

	return(block_Dmatrix,haplotype_block_name_container,haplotype_marker_name_container)


def haplotype_DM_generator(block_index,clutering_algorithm,haplotypes,n_clusters,breakpoints,positions,ch):
	
	dictionary = {}
	r,c = haplotypes.shape

	dedup_haplotypes = np.asarray(haplotypes.drop_duplicates(keep = 'first'))
	d_r,d_c = dedup_haplotypes.shape

	if d_r < n_clusters:  # The situation no.1 no clutering_algorithm is necessary
		
		for i in range(d_r):
			haplotype_ = "".join(map(str,dedup_haplotypes[i,:]))
			dictionary[haplotype_] = i
		haplotype_names = [ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])+'_'+str(l) for l in range(d_r)]
		block_name = ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])
		marker_name = ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])

		DM_matrix_1 = np.zeros((int(r/2),d_r),dtype=int)
		DM_matrix_2 = np.zeros((int(r/2),d_r),dtype=int)
	
	else: # The situation no.2 Too many haplotyps, clutering_algorithm is needed.
		# choose the clustering function
		if clutering_algorithm == 'xmeans':
			clusters = xmeans_clustering(list(dedup_haplotypes))
		elif clutering_algorithm == 'affinity_propagation':
			clusters = affinity_propagation_update(dedup_haplotypes)
			if clusters[0] == -1 or max(clusters) == 0:
				clusters = np.arange(d_r)
				print("Affinity propagation failed",clusters)
		elif clutering_algorithm == 'gmeans':
			clusters = gmeans_clustering(list(haplotypes.values))
		
		elif clutering_algorithm == 'DBSCAN':
			clusters = DBSCAN_clustering(dedup_haplotypes)
			if len(np.unique(clusters)) == 1:
				print("failed")
				clusters = np.arange(d_r)
		elif clutering_algorithm == 'KNN':
			clusters = KNN_Spectral(dedup_haplotypes)

		elif clutering_algorithm == 'local':
			clusters = local_scale_Spectral(dedup_haplotypes)
		elif clutering_algorithm == 'modularity':
			clusters = local_scale_modularity(dedup_haplotypes)

		# generate the dictionary. key: haplotype (str), values (cluster index)
		for j in range(d_r):
			haplotype_ = "".join(map(str,dedup_haplotypes[j,:]))
			dictionary[haplotype_] = clusters[j]
		#the column names of the dataframe
		haplotype_names = [ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])+'_'+str(i) for i in range(max(clusters)+1)]
		block_name = ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])
		marker_name = ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])

		DM_matrix_1 = np.zeros((int(r/2),max(clusters)+1),dtype=int)
		DM_matrix_2 = np.zeros((int(r/2),max(clusters)+1),dtype=int)

	# generate the haplotype design matrix
	for k in range(int(r/2)):
		tmp_1 = "".join(map(str,haplotypes.values[k,:]))
		tmp_2 = "".join(map(str,haplotypes.values[k+int(r/2),:]))
		l_1 = dictionary[tmp_1]
		l_2 = dictionary[tmp_2]
		DM_matrix_1[k,l_1] += 1
		DM_matrix_2[k,l_2] += 1

	haplotype_DM = DM_matrix_1 + DM_matrix_2

	if haplotype_DM.shape[1] == 2:
		haplotype_DM_minus1 = haplotype_DM[:,1]
		haplotype_names = [haplotype_names[1]]
	else:
		hap_freq = np.sum(haplotype_DM,0) / (haplotype_DM.shape[0]*2)
		hap_freq_order = np.argsort(hap_freq)
		haplotype_DM_minus1 = haplotype_DM[:,hap_freq_order[1:]]
		haplotype_names = haplotype_names[1:]
	haplotype_DM_minus1 = pd.DataFrame(haplotype_DM_minus1,columns=haplotype_names)	

	return(haplotype_DM_minus1,block_name,haplotype_names,marker_name)

def format_bimbam(haplotype_Dmatrix,haplotype_names):
	r,c = haplotype_Dmatrix.shape
	l = len(haplotype_names)
	if r != l:
		print("haplotype matrix is not built right!!!!")
	else:
		columns = ['haplotype_names','hap','no-hap']
		bimbam_ = pd.DataFrame(index=range(l),columns=columns)
		bimbam_['haplotype_names'] = haplotype_names
		bimbam_['hap'] = '-'
		bimbam_['no-hap'] = '-'
		bimbam = pd.concat([bimbam_,pd.DataFrame(haplotype_Dmatrix)],axis=1,ignore_index=True)
	return(bimbam)


def cat(dictionary,keys):
	dict_cat = []
	for i in keys:
		for j in dictionary[i]:
			dict_cat.append(j)
	return(dict_cat)



def block_pip_calculation(gamma_trace,block_haplotypes,block_positions):

	nrow = gamma_trace.shape[0] ## number of MCMC iterations
	ncol = len(block_haplotypes)
	block_gamma = np.zeros(shape = (nrow,ncol))
	for i in range(len(block_positions)):
		col_index = block_haplotypes[block_positions[i]]
		x = np.sum(gamma_trace[:,col_index],axis = 1)
		row_index = np.where(x >= 1)
		block_gamma[row_index[0],i] = 1
	block_pip = np.mean(block_gamma,axis = 0)
	return(block_pip)



def block_gamma(gamma,block_haplotypes,block_positions):

	block_gamma = np.zeros(len(block_positions))
	for i in range(len(block_positions)):
		index = block_haplotypes[block_positions[i]]
		block_sum = np.sum(gamma[index])

		if block_sum >= 1:
			block_gamma[i] = 1
	return(block_gamma)

def fdr_calculation(pip):

	ordered_index = np.argsort(pip)[::-1]

	sorted_pip = pip[ordered_index]

	## FDR calculation from sorted pip values

	fdr = []

	for i in range(len(sorted_pip)):
		if i == 0:
			fdr.append( 1 - sorted_pip[i])
		else:
			fdr.append( 1 - np.mean(sorted_pip[:i+1]))

	return(ordered_index,fdr)

def pip_calculation_2(haplotype_pip,block_haplotypes,block_positions):

	block_pip = np.zeros(len(block_haplotypes))

	for i in range(len(block_positions)):
		hap_index = block_haplotypes[block_positions[i]]
		block_pip_ = 1
		for j in hap_index:
			block_pip_ = block_pip_ * (1-haplotype_pip[j])
		block_pip[i] = 1 - block_pip_

	return(block_pip)

def pip_calculation_max(haplotype_pip,block_haplotypes,block_positions):

	block_pip = np.zeros(len(block_haplotypes))

	for i in range(len(block_positions)):
		hap_index = block_haplotypes[block_positions[i]]
		block_pip[i] = np.amax(haplotype_pip[hap_index])

	return(block_pip)

def block_beta(beta_mean,block_haplotypes,block_positions):

	block_beta = np.zeros(len(block_haplotypes))

	for i in range(len(block_positions)):
		hap_index = block_haplotypes[block_positions[i]]
		block_beta[i] = np.amax(beta_mean[hap_index])

	return(block_beta)

def merge_welford(A_mean, A_M2,A_n,B_mean,B_M2,B_n):

    if A_n == 0:
        return(B_mean,B_M2,B_n)

    if B_n == 0:
        return(A_mean,A_M2,A_n)
    
    n_new = A_n + B_n
    delta = B_mean - A_mean

    mean = A_mean + delta * (B_n / n_new)
    M2 = A_M2 + B_M2 + (delta * delta) * (A_n * B_n / n_new)

    return(mean,M2,n_new)

def welford(mean,M2,x,n):
    n = n + 1

    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    M2 += delta * delta2

    return(mean,M2)

def convergence_geweke_test(trace,top5_beta_trace,start,end):
    max_z = []

    ## convergence for the trace values
    n = trace.shape[1]
    for t in range(n):
        trace_convergence = trace[start:end,t]
        trace_t_convergence_zscores = geweke.geweke(trace_convergence)[:,1]
        max_z.append(np.amax(np.absolute(trace_t_convergence_zscores)))

    m = top5_beta_trace.shape[1]
    for b in range(m):
        top_beta_convergence = top5_beta_trace[start:end,b]
        beta_b_convergence_zscores = geweke.geweke(top_beta_convergence)[:,1]
        max_z.append(np.amax(np.absolute(beta_b_convergence_zscores)))

    if np.amax(max_z) < 1.5:
        return(1)






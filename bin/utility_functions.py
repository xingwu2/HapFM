
import re
import numpy as np
import pandas as pd
import scipy
import networkx as nx

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

'''
define functions
'''

''' 

preprocessing steps:

1. convert vcf to SNP matrix (sum of two haplotype matrix)
2. LD measurement
3. independent LD block partition  

'''

def vcf2hapmatrix(vcf):
	hap_matrix_d1 = {} #haplotype 1 of individuals, key as chromosome number
	hap_matrix_d2 = {} #haplotype 2 of individuals, key as chromosome number
	variant_names = {}
	variant_positions = {} #key as chromosome number
	chromosome = [] #key as chromosome number and value as number of SNPs per chromosome
	
	with open(vcf,"r") as VCF:
		for line in VCF:
			if re.search("^##",line): ## skip the first annotation lines
				continue
			elif re.search("^#CHROM",line): ## acquire the sample name information
				line = line.strip("\n")
				ind_names = line.split("\t")[9:]
			else:
				line = line.strip("\n")
				items = line.split("\t")
				ch = items[0]

				if ch not in chromosome:
					chromosome.append(ch)
					variant_names[ch] = [items[2]]
					variant_positions[ch] = [int(items[1])]
					hap_matrix_d1[ch] = []
					hap_matrix_d2[ch] = []
					genotype = items[9:]
					for i in range(len(genotype)):
						m = re.search('([0-9])\|([0-9])',genotype[i])
						hap_matrix_d1[ch].append(int(m.group(1)))
						hap_matrix_d2[ch].append(int(m.group(2)))
				else:
					variant_names[ch].append(items[2])
					variant_positions[ch].append(int(items[1]))
					genotype = items[9:]
					for i in range(len(genotype)):
						m = re.search('([0-9])\|([0-9])',genotype[i])
						hap_matrix_d1[ch].append(int(m.group(1)))
						hap_matrix_d2[ch].append(int(m.group(2)))

	for ch in chromosome:
		hap_matrix_d1[ch] = np.reshape(np.asarray(hap_matrix_d1[ch],dtype=int),(len(variant_names[ch]),len(ind_names)))
		hap_matrix_d2[ch] = np.reshape(np.asarray(hap_matrix_d2[ch],dtype=int),(len(variant_names[ch]),len(ind_names)))

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
		if gw_breakpoints[j][1] - gw_breakpoints[j][0] < 500:
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
		if gw_breakpoints[j][1] - gw_breakpoints[j][0] < 6:

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
	# number of clusters that can be allocated is 20.
	xmeans_instance = xmeans(array, initial_centers, 10)
	xmeans_instance.process()
	# Extract clustering results: clusters and their centers
	clusters_ = xmeans_instance.get_clusters()
	clusters = [0]*len(array)
	for i in range(len(clusters_)):
		for j in clusters_[i]:
			clusters[j] = i
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


def DBSCAN_clustering(np_array):
	r,c = np_array.shape
	clusters = DBSCAN(eps=0.1,min_samples=2,p=2).fit(np_array)
	labels = clusters.labels_
	return(labels)

def local_scale_Spectral(np_array):
	r,c =np_array.shape
	k = max(int(r/10),10)

	dists = squareform(pdist((np_array)))
	knn_distances = np.sort(dists, axis=0)[k]
	knn_distances = knn_distances[np.newaxis].T
	local_scale = knn_distances.dot(knn_distances.T)
	affinity_matrix = - dists * dists / local_scale
	affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
	affinity_matrix = np.exp(affinity_matrix)
	np.fill_diagonal(affinity_matrix, 0)

	L = csgraph.laplacian(affinity_matrix,normed = True)
	eig_val, eig_vec = np.linalg.eig(L)
	eig_val = np.real(eig_val)
	eig_vec = np.real(eig_vec)
	
	eig_vec = eig_vec[:,np.argsort(eig_val)]
	eig_val = eig_val[np.argsort(eig_val)]

	if sum(np.iscomplex(eig_val)) > 0:
		print("Spectral Clustering failed. Clusters are assigned by affinity_propagation.")
		print(np_array.shape)
		labels = affinity_propagation(np_array)
		print(max(labels))
		if labels[0] == -1 or max(labels) == 0:
			labels = np.arange(np_array.shape[0])
			print("Affinity propagation failed")
		
	else:
		index_largest_gap = np.argsort(np.diff(eig_val))[::-1][0]
		#print(index_largest_gap)
		n_clusters = index_largest_gap + 2
		V = eig_vec[:,:n_clusters]
		Z = linkage(V, 'ward')
		labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
	return(labels)


def local_scale_modularity(df):
	r,c = df.shape
	W = np.zeros(shape=(r,r))
	k = max(int(r/10),5)
	dists = squareform(pdist(df))

	knn_distances = np.sort(dists, axis=0)[k]
	knn_distances = knn_distances[np.newaxis].T
	local_scale = knn_distances.dot(knn_distances.T)
	W = - dists * dists / local_scale

	W[np.where(np.isnan(W))] = 0.0
	W = np.exp(W)
	np.fill_diagonal(W, 0)

	connectivity = kneighbors_graph(X=df, n_neighbors=k, mode='connectivity')
	connectivity = 0.5 * (connectivity + connectivity.T) ## make connectivity symmetric

	connectivity_matrix = connectivity.toarray()
	weighted_connectivity = np.multiply(W,connectivity_matrix)
	G = nx.from_numpy_array(weighted_connectivity)
	## this is a more stable function that the results can be reproduced due to the seed setting
	louvain_partition = nx.community.louvain_communities(G,seed=k)
	labels = np.zeros(r,dtype=int)
	for i in range(r):
		for k in range(len(louvain_partition)):
			clusters_ = [*louvain_partition[k],]
			for m in clusters_:
				labels[m] = k
	return(labels)



def Spectral_clustering(np_array):
	r,c =np_array.shape
	dists = squareform(pdist((np_array)))

	W = np.zeros(shape=(r,r))

	for i in range(r):
		for j in range(i+1,r):
			W[i,j] = np.exp(-(dists[i,j]**2))
			W[j,i] = W[i,j]

	D = np.diag(np.sum(W,axis=0))

	D_inv = np.linalg.inv(D)

	L_rw = np.identity(r) - np.matmul(D_inv,W)

	eig_val, eig_vec = np.linalg.eig(L_rw)

	eig_vec = eig_vec[:,np.argsort(eig_val)]
	eig_val = eig_val[np.argsort(eig_val)]
	
	if sum(np.iscomplex(eig_val)) > 0:
		print("Spectral Clustering failed. Clusters are assigned by affinity_propagation.")
		print(np_array.shape)
		labels = affinity_propagation(np_array,weights)
		if labels[0] == -1 or max(labels) == 0:
			labels = np.arange(np_array.shape[0])
			print("Affinity propagation failed")
	else:	
		index_eigen_gap = np.argmax(np.diff(eig_val))
		n_clusters = index_eigen_gap + 2
		V = eig_vec[:,:n_clusters]
		Z = linkage(V, 'ward')
		labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

	return(labels)

def KNN_Spectral(df):
	r,c =df.shape
	k = max(int(r/10),5)
	connectivity = kneighbors_graph(X=df, n_neighbors=k, mode='distance')

	A = (1/2)*(connectivity + connectivity.T)
	
	L = csgraph.laplacian(A,normed = True)

	L = L.toarray()

	eig_val, eig_vec = np.linalg.eig(L)

	eig_vec = eig_vec[:,np.argsort(eig_val)]
	eig_val = eig_val[np.argsort(eig_val)]
	#print(eig_val)
	
	if sum(np.iscomplex(eig_val)) > 0:
		print("Spectral Clustering failed. Clusters are assigned by affinity_propagation.")
		labels = affinity_propagation(df)
		print(max(labels))
		if labels[0] == -1 or max(labels) == 0:
			labels = np.arange(df.shape[0])
			print("Affinity propagation failed")
	else:	
		index_eigen_gap = np.argmax(np.diff(eig_val))
		n_clusters = index_eigen_gap + 2
		V = eig_vec[:,:n_clusters]
		Z = linkage(V, 'ward')
		labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
		
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
		if breakpoints[block_index][0] == 0:
			haplotype_names = [ch+"@"+str(0)+"-"+str(positions[breakpoints[block_index][1]+1])+'_'+str(l) for l in range(d_r)]
			block_name = ch+"@"+str(0)+"-"+str(positions[breakpoints[block_index][1]+1])
			marker_name = ch+"@"+str(positions[breakpoints[block_index][0]]) + "-" + str(positions[breakpoints[block_index][1]])
		elif breakpoints[block_index][1] == len(positions):
			haplotype_names = [ch+"@"+str(positions[breakpoints[block_index][0]-1])+"-"+str(positions[breakpoints[block_index][1]])+'_'+str(l) for l in range(d_r)]
			block_name = ch+"@"+str(positions[breakpoints[block_index][0]-1])+"-"+str(positions[breakpoints[block_index][1]])
			marker_name = ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])
		else:
			haplotype_names = [ch+"@"+str(positions[breakpoints[block_index][0]-1])+"-"+str(positions[breakpoints[block_index][1]+1])+'_'+str(l) for l in range(d_r)]
			block_name = ch+"@"+str(positions[breakpoints[block_index][0]-1])+"-"+str(positions[breakpoints[block_index][1]+1])
			marker_name = ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])

		DM_matrix_1 = np.zeros((int(r/2),d_r),dtype=int)
		DM_matrix_2 = np.zeros((int(r/2),d_r),dtype=int)
	
	else: # The situation no.2 Too many haplotyps, clutering_algorithm is needed.
		# choose the clustering function
		if clutering_algorithm == 'xmeans':
			clusters = xmeans_clustering(list(dedup_haplotypes))
		elif clutering_algorithm == 'affinity_propagation':
			clusters = affinity_propagation(dedup_haplotypes)
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
		if breakpoints[block_index][0] == 0:
			haplotype_names = [ch+"@"+str(0)+"-"+str(positions[breakpoints[block_index][1]+1])+'_'+str(l) for l in range(max(clusters)+1)]
			block_name = ch+"@"+str(0)+"-"+str(positions[breakpoints[block_index][1]+1])
			marker_name = ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+ str(positions[breakpoints[block_index][1]])
		
		elif breakpoints[block_index][1] == len(positions) - 1:
			haplotype_names = [ch+"@"+str(positions[breakpoints[block_index][0]-1])+"-"+str(positions[breakpoints[block_index][1]])+'_'+str(i) for i in range(max(clusters)+1)]
			block_name = ch+"@"+str(positions[breakpoints[block_index][0]-1])+"-"+str(positions[breakpoints[block_index][1]])
			marker_name = ch+"@"+str(positions[breakpoints[block_index][0]])+"-"+str(positions[breakpoints[block_index][1]])
		
		else:
			haplotype_names = [ch+"@"+str(positions[breakpoints[block_index][0]-1])+"-"+str(positions[breakpoints[block_index][1]+1])+'_'+str(i) for i in range(max(clusters)+1)]
			block_name = ch+"@"+str(positions[breakpoints[block_index][0]-1])+"-"+str(positions[breakpoints[block_index][1]+1])
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



def pip_calculation_1(haplotype_burnt_gamma,block_haplotypes,block_positions):

	nrow = haplotype_burnt_gamma.shape[0]
	ncol = len(block_haplotypes)
	block_gamma = np.zeros(shape = (nrow,ncol))
	for i in range(len(block_positions)):
		col_index = block_haplotypes[block_positions[i]]
		x = np.sum(haplotype_burnt_gamma[:,col_index],axis = 1)
		row_index = np.where(x >= 1)
		block_gamma[row_index[0],i] = 1
	block_pip = np.mean(block_gamma,axis = 0)
	return(block_pip)


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





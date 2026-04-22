from __future__ import division
from collections import Counter
import numpy as np
from scipy.special import rel_entr

# FAIRNESS METRICS

# CLUSTERING
def create_clusters(sens_column, labels):
    keys = np.unique(labels).tolist()
    clusters = {key: [] for key in keys}
    for i, l  in enumerate(labels):
        # if l not in clusters: clusters[l] = []
        clusters[l].append(sens_column[i])
    return clusters

"""
def max_fairness_cost(dataset, sensitive, labels):
    groups = dataset[sensitive].unique()
    indices = dataset[sensitive]
    
    clusters = create_clusters(indices, labels)
    
    n = dataset.shape[0]
    ideal_props = [((Counter(indices)[g])/n) for g in groups]
    
    MFC_C = []
    for c in clusters.keys():
        MFC_C_G = []
        for i, g in enumerate(groups):        
            count_g = Counter(clusters[c])[g]
            P_g = count_g/(len(clusters[c]))
            d = abs(ideal_props[i]- P_g)
            MFC_C_G.append(d)
        MFC_C.append(MFC_C_G)
    return MFC_C

def entropy(value):
    h = - value * np.log(value + 1e-5)
    return h

def balance_entropy(dataset, sensitive, labels):
    groups = dataset[sensitive].unique()
    d_s = dataset[sensitive]
    clusters = create_clusters(d_s, labels)
    
    entropy_groups = []
    for g in groups:
        entropy_clusters = []
        for c in clusters.keys():
            r_i_f = Counter(clusters[c])[g]/len(clusters[c])
            h = entropy(r_i_f)
            entropy_clusters.append(h)
        entropy_groups.append(sum(entropy_clusters))
    return entropy_groups
"""


"""
    Balance [Chierichetti et al.]
    - The proportion of protected group 0 in cluster must be equal to the proporion of protected group 1 in cluster, for each cluster.
    - 2 or 3 protected groups 
    
    Input
    
    - sensitive, str
        column of sensitive feature
        
    - labels, numpy array
        predicted clustering
        
    Output
    - balance_clusters, numpy matrix
        return min balance
"""

# Balance [Chierichetti] with 3 groups
def balance_chierichetti_3groups(cluster_assign, K):
    
    dimensions = list(cluster_assign.keys())
    S_k = []  # balance of each k cluster
    balance = 0  # min (S_k)

    for k in dimensions:
        cnt_j_0 = 0
        cnt_j_1 = 0
        cnt_j_2 = 0
        
        cnt = 0
        
        for each in cluster_assign[k]:
            if int(each) == 1:
                cnt_j_1 += 1
            elif int(each) == 0:
                cnt_j_0 += 1
            elif int(each) == 2:
                cnt_j_2 += 1
                
            cnt += 1

        if cnt_j_0 != 0 and cnt_j_1 != 0 and cnt_j_2 != 0:
            S_k.append(min([cnt_j_0 / cnt_j_1, cnt_j_1 / cnt_j_0, cnt_j_1 / cnt_j_2 , cnt_j_2 / cnt_j_1 , cnt_j_0 / cnt_j_2, cnt_j_2 / cnt_j_0 ]))
        elif cnt_j_0 == 0 or cnt_j_1 == 0 or cnt_j_2 == 0:
            S_k.append(0)

    balance = min(S_k)
    return balance


# Balance [Chierichetti] with 2 groups
def balance_chierichetti_2groups(cluster_assign, K):
    
    dimensions = list(cluster_assign.keys())
    S_k = []  # balance of each k cluster
    balance = 0  # min (S_k)
    
    for k in dimensions:
        cnt_j_0 = 0
        cnt_j_1 = 0
        cnt = 0
        
        for each in cluster_assign[k]:
            if int(each) == 1:
                cnt_j_1 += 1
            elif int(each) == 0:
                cnt_j_0 += 1
                
            cnt += 1

        if cnt_j_0 != 0 and cnt_j_1 != 0:
            S_k.append(min([ cnt_j_0 / cnt_j_1, cnt_j_1 / cnt_j_0 ]))
        elif cnt_j_0 == 0 or cnt_j_1 == 0:
            S_k.append(0)

    balance = min(S_k)
    return balance


# Balance [Chierichetti] - group 0 and group 1 have the same ratio in the cluster
def balance_chierichetti(sensitive, labels):
    cluster_assign = create_clusters(sensitive, labels)
    K = len(cluster_assign.keys())
    num_groups = len(np.unique(sensitive))
    
    if num_groups == 2:
        return balance_chierichetti_2groups(cluster_assign, K)
    elif num_groups == 3:
        return balance_chierichetti_3groups(cluster_assign, K)
    else:
        raise ValueError("The number of protected groups must be 2 or 3.")


"""
    Balance [Bera et al.]
    - the color (protected group) proportion of each cluster must be similar to that in the original data.
    - multiple protected groups 
    
    Input
    
    - sensitive, str
        column of sensitive feature
        
    - labels, numpy array
        predicted clustering
        
    Output
    - balance_clusters, numpy matrix
        return min balance
"""
#def balance_gen(dataset, sensitive, labels, delta=0.2):
def balance_gen(sensitive, labels):
    
    indices = sensitive
    groups = np.unique(indices)    
    clusters = create_clusters(indices, labels)

    #fairness_clusters = []
    balance_clusters = []
    
    for c in clusters.keys():
        balance_groups = []
        #fairness_groups = []
        for g in groups:
            r_i = Counter(indices)[g]/sensitive.shape[0]
            r_i_f = Counter(clusters[c])[g]/len(clusters[c])
            if r_i != 0 and r_i_f !=0: balance_c = min(r_i/r_i_f, r_i_f/r_i)
            else: balance_c = 0.0
            balance_groups.append(balance_c)
            #beta_i = r_i*(1-delta) #lb
            #alpha_i = r_i/(1-delta)
            #print(beta_i, alpha_i)
        balance_clusters.append(balance_groups)
    return np.min(balance_clusters)


# Ziko et al. Variational Fair Clustering:
# Kullback-Leibler divergence between the required protected group proportion tau (tau=1/k) 
# and achieved proportion within the clusters
def KL_fairness_error(cluster_assign, K, sensitive):
    
    if not isinstance(cluster_assign, np.ndarray):
        cluster_assign = np.array(cluster_assign)

    if not isinstance(sensitive, np.ndarray):
        sensitive = np.array(sensitive)

    cnt = sensitive.shape[0]
    protected_groups, cnt_j = np.unique(sensitive, return_counts=True)
    U = cnt_j / cnt         # distribution of each protected group in original target dataset for each group j
    P_k_sum_over_j = []     # distribution in kth cluster  sum_k( sum_j(   Uj * j wale/total_in_cluster ) )

    clusters, cnt_total = np.unique(cluster_assign, return_counts=True)
    
    for idx, k in enumerate(clusters):
        for j in protected_groups:
            cnt_j_cluster = np.count_nonzero((cluster_assign == k) & (sensitive == j))
            div = cnt_j_cluster / cnt_total[idx]
            KL_fair = rel_entr(U[j], div)
            P_k_sum_over_j.append(KL_fair)

            if np.isinf(KL_fair):
                if U[j] < 0 or div < 0:
                    #print(f"U[j] = {U[j]}")
                    #print(f"div = {div}")
                    raise ValueError("KL fair = inf and (Uj < 0 or div < 0)")

    f_error = np.sum(P_k_sum_over_j)
    return f_error

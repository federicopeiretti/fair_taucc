import sys
import os
import pandas as pd
import numpy as np
import logging as l
import datetime
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
#from algorithms.coclust_DP import CoClust as CoClust_DP
#from algorithms.coclust_incremental2 import CoClust

"""
def execute_test_dp(f, V, results, noise=0, eps = 1, init = [], n_iterations=4, verbose = False):
    '''
    Execute CoClust algorithm and write an output file (already existing and open)

    Parameters:
    ----------

    f: output file (open). See CreateOutputFile for a description of the fields.      
    V: tensor
    model: 'CoClust' or 'CC'
    x: target on mode 0
    y: target on mode 1
    z: target on mode 2
    noise: only for synthetic tensors. Amount of noise added to the perfect tensor
    sparsity: sparsity of the tensor (number of entries != 0 / total number of entries)

    '''
    if len(init)!=2:
        raise ValueError('init must be a list of two integers')
    model = CoClust_DP(eps = eps, n_iterations = n_iterations, k = init[0], l = init[1], verbose = verbose)
    model.fit(V)
  
    l_nmi = []
    l_ari = []
    _assignment = [model._row_assignment, model._col_assignment]
    for i in range(len(results)):
        l_nmi.append(nmi(results[i], _assignment[i], average_method='arithmetic'))
        l_ari.append(ari(results[i], _assignment[i]))


    n = ','.join(str(e) for e in l_nmi)
    a = ','.join(str(e) for e in l_ari)
    init_clusters_x = model.k
    init_clusters_y = model.l

    f.write(f"{eps}, {V.shape[0]}, {V.shape[1]}, {len(np.unique(results[0]))}, {len(np.unique(results[1]))},{noise},{model.tau_x[-1]},{model.tau_y[-1]},{n},{a},{model._n_row_clusters},{model._n_col_clusters},{model.execution_time_},{model._actual_n_iterations},{init_clusters_x},{init_clusters_y},{n_iterations}\n")
    if verbose:
        return model

def execute_test_cc(f, V, results, noise=0, init = [], n_iterations=4, verbose = False):
    '''
    Execute CoClust algorithm and write an output file (already existing and open)

    Parameters:
    ----------

    f: output file (open). See CreateOutputFile for a description of the fields.      
    V: tensor
    model: 'CoClust' or 'CC'
    x: target on mode 0
    y: target on mode 1
    z: target on mode 2
    noise: only for synthetic tensors. Amount of noise added to the perfect tensor
    sparsity: sparsity of the tensor (number of entries != 0 / total number of entries)

    '''
    if len(init)!=2:
        raise ValueError('init must be a list of two integers')
    model = CoClust(k = init[0], l = init[1], verbose = verbose)
    model.fit(V)
  
    l_nmi = []
    l_ari = []
    _assignment = [model._row_assignment, model._col_assignment]
    for i in range(len(results)):
        l_nmi.append(nmi(results[i], _assignment[i], average_method='arithmetic'))
        l_ari.append(ari(results[i], _assignment[i]))


    n = ','.join(str(e) for e in l_nmi)
    a = ','.join(str(e) for e in l_ari)
    init_clusters_x = model.k
    init_clusters_y = model.l

    f.write(f"cc, {V.shape[0]}, {V.shape[1]}, {len(np.unique(results[0]))}, {len(np.unique(results[1]))},{noise},{model.tau_x[-1]},{model.tau_y[-1]},{n},{a},{model._n_row_clusters},{model._n_col_clusters},{model.execution_time_},{model._actual_n_iterations},{init_clusters_x},{init_clusters_y},{n_iterations}\n")
    if verbose:
        return model
"""

def CreateOutputFile(partial_name, own_directory = False, date = True, overwrite = False):
    '''
    Create and open a file containing the header described below.

    Parameters:
    ----------
    partial_name: partial name of the file and the directory that will contain the file.
    own_directory: boolean. Default: False.
        If true, a new directory './output/_{partial_name}/aaaa-mm-gg_hh.mm.ss' will be created.
        If flase, the path of the file will be './output/_{partial_name}'.
    date: boolean. Default: True.
        If true, the file name will include datetime.
        If false, it will not.
    overwrite: boolean. Default: False.
        If true, overwrite the existent file (if there exists a file with the same name)
        If false, append the new results.
                

    Output
    ------
    f: file (open). Each record contains the following fields, separated by commas (csv file):
        - model: 'CoClust' or 'CC'
        - dim_x: dimension of the tensor on mode 0
        - dim_y: dimension of the tensor on mode 1
        - x_num_classes: correct number of clusters on mode 0
        - y_num_classes: correct number of clusters on mode 1
        - noise: only for synthetic tensors. Amount of noise added to the perfect tensor
        - tau_x: final tau_{x|y}
        - tau_y: final tau_{y|x}
        - nmi_x: normalized mutual information score on mode 0
        - nmi_y: normalized mutual information score on mode 1
        - ari_x: adjusted rand index on mode 0
        - ari_y: adjusted rand index on mode 1
        - x_num_clusters: number of clusters on mode 0 detected by CoClust
        - y_num_clusters: number of clusters on mode 1 detected by CoClust
        - execution time
        - iter: total number of iterations
        - init_clusters_x: number of initial clusters on mode 0
        - init_clusters_y: number of initial clusters on mode 1

        File name:{partial_name}_aaaa-mm-gg_hh.mm.ss.csv or {partial_name}_results.csv
    dt: datetime (as in the directory/ file name)

    
    '''

    
    dt = f"{datetime.datetime.now()}"
    if own_directory:
        data_path = f"./output/_{partial_name}/" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + "/"
    else:
        data_path = f"./output/_{partial_name}/"
    directory = os.path.dirname(data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    new = True
    if date:
        file_name = partial_name + "_" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + ".csv"
    else:
        file_name = partial_name + '_results.csv'
        if os.path.isfile(data_path + file_name):
            if overwrite:
                os.remove(data_path + file_name)
            else:
                new = False
            
            
    f = open(data_path + file_name, "a",1)
    if new:
        f.write("model,dim_x,dim_y,x_num_classes,y_num_classes,noise,tau_x,tau_y,nmi_x,nmi_y,ari_x,ari_y,x_num_clusters,y_num_clusters,execution_time,iter, init_clusters_x,init_clusters_y,n_iterations\n")
        
    return f, dt


def CreateLogger(input_level = 'INFO'):
    level = {'DEBUG':l.DEBUG, 'INFO':l.INFO, 'WARNING':l.WARNING, 'ERROR':l.ERROR, 'CRITICAL':l.CRITICAL}
    logger = l.getLogger()
    logger.setLevel(level[input_level])

    return logger


# This function check if the path already exists, otherwise create the directory.
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created successfully.")
    else:
        print(f"Folder '{path}' already exists.")


def generate_Sx_from_target(target_r, num_groups, group_prob):
    if (np.sum(group_prob) != 1):
        raise ValueError(f"The probability must be in [0,1] and the sum must be equal to 1")
    if (group_prob.shape[0] != num_groups):
        raise ValueError(f"The size of group_prob must be equal to num_groups")
    
    balanced_array = np.zeros_like(target_r)
    
    for cluster in np.unique(target_r):
        index = np.where(target_r == cluster)
        balanced_array[index] = np.random.choice(np.arange(num_groups), size=target_r[index].shape, p=group_prob)
    
    return balanced_array


def list_to_set(lst):
    return set(lst)

# SEE get_aggregated()
def aggregated_results_fair(dataset, sensitive, num_groups, algorithm="taucc_fair_max", init="random", root=""):

    path = f"{root}/results/{dataset}/{sensitive}/{algorithm}/init_{init}"
    df = pd.read_csv(path + "/results_runs.csv", sep=";")
    df.drop(["run", "num_iter", "row_clus", "col_clus"], axis=1, inplace=True)

    if num_groups == 2:
        df_mean = df.groupby(["fair_majority", "fair_minority"]).mean()
        df_std = df.groupby(["fair_majority", "fair_minority"]).std()
        df_var = df.groupby(["fair_majority", "fair_minority"]).var()
    else:
        df_mean = df.groupby(["fair_majority", "fair_minority1", "fair_minority2"]).mean()
        df_std = df.groupby(["fair_majority", "fair_minority1", "fair_minority2"]).std()
        df_var = df.groupby(["fair_majority", "fair_minority1", "fair_minority2"]).var()

    aggregated = pd.DataFrame()
    for key in df_mean.keys():
        aggregated[f"{key}_mean"] = df_mean[key].values
    for key in df_std.keys():
        aggregated[f"{key}_std"] = df_std[key].values
    for key in df_var.keys():
        aggregated[f"{key}_var"] = df_var[key].values

    fair_major = [item[0] for item in list(df_mean.index)]
    fair_minor1 = [item[1] for item in list(df_mean.index)]
    aggregated.insert(0, "fair_majority", fair_major)
    aggregated.insert(1, "fair_minority", fair_minor1)

    if num_groups == 3:
        fair_minor2 = [item[2] for item in list(df_mean.index)]
        aggregated.insert(2, "fair_minority2", fair_minor2)

    aggregated.to_csv(path + "/aggregated.csv", index=False)


# SEE get_aggregated()
def aggregated_results_vanilla(dataset, sensitive, num_run=10, algorithm="taucc_vanilla", init="random", root=""):
    path = f"{root}/results/{dataset}/{sensitive}/{algorithm}/init_{init}"
    df = pd.read_csv(path + "/results_runs.csv", sep=";")
    cols_metrics = df.columns.tolist()[4:]
    stats_vanilla = {}

    for metric in cols_metrics:
        values = df[metric].tolist()[:num_run]
        stats_vanilla[f"{metric}_mean"] = [np.mean(values)]
        stats_vanilla[f"{metric}_std"] = [np.std(values)]
        stats_vanilla[f"{metric}_var"] = [np.var(values)]

    df_stats = pd.DataFrame(stats_vanilla)
    df_stats.to_csv(path + f"/aggregated.csv", index=False)


def get_aggregated(df, group_keys, exclude_cols, path=None, filename=None, save=False):
    """
    Aggregate results of a dataframe with several runs
        @param df: dataframe (results_runs.csv)
        @param group_keys: columns to be group (array)
        @param exclude_cols: columns to be excluded by the aggregation
        @param path: path in which csv is saved
        @param filename: csv filename
        @param save: True (saved), False (only return)
        @return aggregated results (dataframe)
    """
    exclude_cols = exclude_cols | set(group_keys)
    metrics = [c for c in df.columns if c not in exclude_cols]

    agg_df = (
        df.groupby(group_keys)[metrics]
        .agg(["mean", "std", "var"])
    )

    agg_df.columns = ["_".join(col) for col in agg_df.columns]
    agg_df = agg_df.reset_index()
    
    if save:
        agg_df.to_csv(f"{path}/{filename}.csv", index=False)
    
    return agg_df


def get_aggregated_synthetic(clusters, groups, algorithm, path, save):
    
    exclude_cols = {"run", "num_iter", "row_clus", "col_clus"}

    if algorithm == "taucc_vanilla":
        separator = ";"
        group_keys = ["sensitive_p"]

    elif algorithm == "taucc_fair_max":
        separator = ","
        if groups == 2:
            group_keys = ["sensitive_p", "fair_majority", "fair_minority"]
        else:
            group_keys = ["sensitive_p", "fair_majority", "fair_minority1", "fair_minority2"]
    else:
        raise Exception("Exception")

    df = pd.read_csv(path + f"/results_runs_groups{groups}.csv", sep=separator)
    df.drop("num_groups", axis=1, inplace=True)
    filename = f"aggregated_groups{groups}"
    result = get_aggregated(group_keys, exclude_cols, path, filename, save=True)
    return result



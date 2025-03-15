"""
    Experiment 1

    Fair TauCC v1
    Sensitive attribute about rows
    with 3 protected groups

    fair_parameters in [0,1]:
    - majority: [0.0, 1.0] with step=0.10
    - minority1: [0.0, 1.0] with step=0.10
    - minority2: [0.0, 1.0] with step=0.10

    Function: (|group| / num_clusters) * fair_parameters

    *** MovieLens 1M ***
    users x movies
    values: ratings
    Sx: age
    true labels: genres

    Runs: 10
    K = 10
    L = 10
"""

import numpy as np
import pandas as pd
import time
import pickle

import matplotlib.pyplot as plt

from global_var import *

from tauCC.src.taucc.taucc_fair_v3_tau_fairness import FairCoclusRows
from tauCC.src.utils_plot import plot_tau, plot_coclus
from tauCC.src.fairness_metrics import balance_gen, KL_fairness_error, balance_chierichetti
from tauCC.src.utils import create_path

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score


results = {
    "fair_majority": [],
    "fair_minority1": [],
    "fair_minority2": [],
    "run": [],
    "num_iter": [],
    "row_clus": [],
    "col_clus": [],
    "tau_x": [],
    "tau_y": [],
    "NMI_true_labels": [],
    "AMI_true_labels": [],
    "ARI_true_labels": [],
    "NMI_rows": [],
    "AMI_rows": [],
    "ARI_rows": [],
    "NMI_cols": [],
    "AMI_cols": [],
    "ARI_cols": [],
    "balance_chierichetti": [],
    "balance_bera": [],
    "KL_fairness_error": [],
    "time": []
}

results_keys = results.keys()
num_keys = len(results_keys)

if not os.path.exists(PATH_RESULTS_FAIR_INIT + "/results_runs.csv"):
    with open(PATH_RESULTS_FAIR_INIT + "/results_runs.csv", "a") as file:
        for idx, key in enumerate(results_keys):
            file.write(f"{key}")
            if idx == num_keys - 1:
                file.write("\n")
            else:
                file.write(";")


for fair_majority in fair_majority_range:

    PATH_MAJORITY = PATH_RESULTS_FAIR_INIT + f"/majority{fair_majority}"
    create_path(PATH_MAJORITY)

    for fair_min1 in fair_minority_range:

        PATH_MIN1 = PATH_MAJORITY + f"/min1_{fair_min1}"
        create_path(PATH_MIN1)

        for fair_min2 in fair_minority_range2:

            PATH_MIN2 = PATH_MIN1 + f"/min2_{fair_min2}"
            create_path(PATH_MIN2)

            PATH_PLOTS = PATH_MIN2 + "/plots"
            create_path(PATH_PLOTS)
            PATH_PLOT_TAU = PATH_PLOTS + "/tau"
            create_path(PATH_PLOT_TAU)
            PATH_PLOT_COCLUS = PATH_PLOTS + "/coclus"
            create_path(PATH_PLOT_COCLUS)

            # major minor1	minor2	fair_params [0,1,2]
            # 0 	1 	    2	    [major, min1, min2]
            # 0 	2 	    1	    [major, min2, min1]
            # 1 	0 	    2	    [min1, major, min2]
            # 1 	2 	    0	    [min2, major, min1]
            # 2 	0 	    1	    [min1, min2, major]
            # 2 	1 	    0	    [min2, min1, major]

            if count_groups[0] >= count_groups[1] and count_groups[1] >= count_groups[2]:
                fair_params = np.array([fair_majority, fair_min1, fair_min2])
            elif count_groups[0] >= count_groups[2] and count_groups[2] >= count_groups[1]:
                fair_params = np.array([fair_majority, fair_min2, fair_min1])
            elif count_groups[1] >= count_groups[0] and count_groups[0] >= count_groups[2]:
                fair_params = np.array([fair_min1, fair_majority, fair_min2])
            elif count_groups[1] >= count_groups[2] and count_groups[2] >= count_groups[0]:
                fair_params = np.array([fair_min2, fair_majority, fair_min1])
            elif count_groups[2] >= count_groups[0] and count_groups[0] >= count_groups[1]:
                fair_params = np.array([fair_min1, fair_min2, fair_majority])
            elif count_groups[2] >= count_groups[1] and count_groups[1] >= count_groups[0]:
                fair_params = np.array([fair_min2, fair_min1, fair_majority])

            for run in run_range:
                print(f"major = {fair_majority}, minor1 = {fair_min1}, minor2 = {fair_min2}")
                print(f"- run = {run}")

                PATH_RUN = PATH_MIN2 + f"/run{run}"
                create_path(PATH_RUN)

                PATH_DATA = PATH_RUN + f"/data"
                create_path(PATH_DATA)

                start_time = 0
                end_time = 0

                start_time = time.time()

                #if INIT == "vanilla":
                #    model = FairCoclusRows(initialization=INIT, verbose=True, stop_k=10, stop_l=10)
                #else:
                model = FairCoclusRows(initialization=INIT, verbose=True, k=10, l=10)

                model.fit(V, Sx, fair_parameters=fair_params)

                end_time = time.time()
                execution_time = end_time - start_time

                model.saveToNpy(PATH_DATA)

                predict_rows = model.row_labels_
                predict_cols = model.column_labels_

                tau_x = model.tau_x[-1]
                tau_y = model.tau_y[-1]

                if TRUE_LABEL_DIM == "cols":
                    NMI = normalized_mutual_info_score(true_labels, predict_cols)
                    AMI = adjusted_mutual_info_score(true_labels, predict_cols)
                    ARI = adjusted_rand_score(true_labels, predict_cols)
                else:
                    NMI = normalized_mutual_info_score(true_labels, predict_rows)
                    AMI = adjusted_mutual_info_score(true_labels, predict_rows)
                    ARI = adjusted_rand_score(true_labels, predict_rows)

                NMI_rows = normalized_mutual_info_score(vanilla_rows, predict_rows)
                AMI_rows = adjusted_mutual_info_score(vanilla_rows, predict_rows)
                ARI_rows = adjusted_rand_score(vanilla_rows, predict_rows)

                NMI_cols = normalized_mutual_info_score(vanilla_cols, predict_cols)
                AMI_cols = adjusted_mutual_info_score(vanilla_cols, predict_cols)
                ARI_cols = adjusted_rand_score(vanilla_cols, predict_cols)

                balance_bera = balance_gen(Sx, predict_rows)
                balance_chier = balance_chierichetti(Sx, predict_rows)

                predict_rows_np = np.array(predict_rows)
                K = len(np.unique(predict_rows_np))
                fair_error = KL_fairness_error(predict_rows_np, K, Sx)

                predict_cols_np = np.array(predict_cols)
                L = len(np.unique(predict_cols_np))

                num_iter = model._actual_n_iterations

                with open(PATH_RESULTS_FAIR_INIT + "/results_runs.csv", "a") as file:
                    file.write(f"{fair_majority};"
                               f"{fair_min1};"
                               f"{fair_min2};"
                               f"{run};"
                               f"{num_iter};"
                               f"{K};"
                               f"{L};"
                               f"{tau_x};"
                               f"{tau_y};"
                               f"{NMI};"
                               f"{AMI};"
                               f"{ARI};"
                               f"{NMI_rows};"
                               f"{AMI_rows};"
                               f"{ARI_rows};"
                               f"{NMI_cols};"
                               f"{AMI_cols};"
                               f"{ARI_cols};"
                               f"{balance_chier};"
                               f"{balance_bera};"
                               f"{fair_error};"
                               f"{execution_time}\n"
                               )

                #fig = plt.figure()
                #plot_tau(model.tau_x, model.tau_y, num_iter + 1, save=True, path=PATH_PLOT_TAU, filename=f"run{run}")
                #plot_coclus(V, predict_rows_np, predict_cols_np, save=True, path=PATH_PLOT_COCLUS, filename=f"run{run}",
                #            title=rf"Fair Co-clustering ($\alpha_0 = {fair_majority}$, $\alpha_1 = {fair_min1}$, $\alpha_2 = {fair_min2}$)")

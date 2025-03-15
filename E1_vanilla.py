"""
    Experiment 1
    
    Vanilla TauCC

    *** MovieLens 1M ***
    users x movies
    values: ratings
    sensitive Sx:
        gender
            - M: 4331       (0)
            - F: 1709       (1)
        age
            < 35            (0)
            35 <= age < 50  (1)
            > 50            (2)

    true labels: genres

    *** Amazon ***
    reviews x words
    values: counts
    Sx: gender (M, F)
    - 275 M
    - 430 F
    true labels: preferred_words_by_category

    Runs: 10
    K = 10
    L = 10
"""

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

from global_var import *

if INIT == "fair":
    from tauCC.src.taucc.taucc_vanilla_new import CoClust
else:
    from tauCC.src.taucc.taucc import CoClust

from tauCC.src.utils_plot import plot_tau, plot_coclus
from tauCC.src.fairness_metrics import balance_gen, KL_fairness_error, balance_chierichetti
from tauCC.src.utils import create_path

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

PATH_PLOTS = PATH_RESULTS_VANILLA_INIT + "/plots"
PATH_PLOT_TAU = PATH_PLOTS + "/tau"
PATH_PLOT_COCLUS = PATH_PLOTS + "/coclus"

create_path(PATH_PLOTS)
create_path(PATH_PLOT_TAU)
create_path(PATH_PLOT_COCLUS)

results = {
    "run": [],
    "num_iter": [],
    "row_clus": [],
    "col_clus": [],
    "tau_x": [],
    "tau_y": [],
    "NMI": [],
    "AMI": [],
    "ARI": [],
    "balance_chierichetti": [],
    "balance_bera": [],
    "KL_fairness_error": [],
    "time": []
}

results_keys = results.keys()
num_keys = len(results_keys)

if not os.path.exists(PATH_RESULTS_VANILLA_INIT + "/results_runs.csv"):
    with open(PATH_RESULTS_VANILLA_INIT + "/results_runs.csv", "a") as file:
        for idx, key in enumerate(results_keys):
            file.write(f"{key}")
            if idx == num_keys - 1:
                file.write("\n")
            else:
                file.write(";")


for run in run_range:

    PATH_RUN = PATH_RESULTS_VANILLA_INIT + f"/run{run}"
    create_path(PATH_RUN)

    # runX/data:
    # - row_clus, col_clus, tau_x, tau_y 
    PATH_DATA = PATH_RUN + f"/data"
    create_path(PATH_DATA)

    start_time = 0
    end_time = 0

    start_time = time.time()

    model = CoClust(initialization=INIT, verbose=True)
    if INIT == "fair":
        model.fit(V, Sx)
    else:
        model.fit(V)

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

    balance_bera = balance_gen(Sx, predict_rows)
    balance_chier = balance_chierichetti(Sx, predict_rows)

    predict_rows_np = np.array(predict_rows)
    K = len(np.unique(predict_rows_np))
    fair_error = KL_fairness_error(predict_rows_np, K, Sx)

    predict_cols_np = np.array(predict_cols)
    L = len(np.unique(predict_cols_np))

    num_iter = model._actual_n_iterations
    
    with open(PATH_RESULTS_VANILLA_INIT + "/results_runs.csv", "a") as file:
        file.write(f"{run};"
                   f"{num_iter};"
                   f"{K};"
                   f"{L};"
                   f"{tau_x};"
                   f"{tau_y};"
                   f"{NMI};"
                   f"{AMI};"
                   f"{ARI};"
                   f"{balance_chier};"
                   f"{balance_bera};"
                   f"{fair_error};"
                   f"{execution_time}\n"
                   )

    #fig = plt.figure()

    #plot_tau(model.tau_x, model.tau_y, num_iter+1, save=True, path=PATH_PLOT_TAU, filename=f"plot_tau_run{run}")

    #plot_coclus(V, predict_rows_np, predict_cols_np, save=True, path=PATH_PLOT_COCLUS, filename=f"run{run}",
    #            title=rf"Vanilla Co-clustering")




import time
import re

#from tauCC.src.taucc.taucc_fair_rows import FairCoclusRows
#from tauCC.src.taucc.taucc_fair_rows_cols import FairCoclus
#from tauCC.src.taucc.taucc_fair_rows_cols_max import FairCoclus
from tauCC.src.fairness_metrics import balance_gen, KL_fairness_error, balance_chierichetti

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from config_synthetic import *

if ALGO_VERSION == "taucc_fair":
    from tauCC.src.taucc.taucc_fair_rows_cols import FairCoclus
elif ALGO_VERSION == "taucc_fair_max":
    from tauCC.src.taucc.taucc_fair_rows_cols_max import FairCoclus

FILENAME_RESULTS = f"/results_runs_groups{NUM_GROUPS}.csv"

results = {
    "num_groups": [],
    "sensitive_p": [],
    "fair_majority": [],
    "fair_minority": [],
    "run": [],
    "num_iter": [],
    "row_clus": [],
    "col_clus": [],
    "tau_x": [],
    "tau_y": [],
    "NMI_rows": [],
    "AMI_rows": [],
    "ARI_rows": [],
    "NMI_cols": [],
    "AMI_cols": [],
    "ARI_cols": [],
    "NMI_van_rows": [],
    "AMI_van_rows": [],
    "ARI_van_rows": [],
    "NMI_van_cols": [],
    "AMI_van_cols": [],
    "ARI_van_cols": [],
    "balance_chierichetti": [],
    "balance_bera": [],
    "KL_fairness_error": [],
    "time": []
}

results_keys = results.keys()
num_keys = len(results_keys)

if not os.path.exists(PATH_RESULTS_FAIR + FILENAME_RESULTS):
    with open(PATH_RESULTS_FAIR + FILENAME_RESULTS, "a") as file:
        for idx, key in enumerate(results_keys):
            file.write(f"{key}")
            if idx == num_keys - 1:
                file.write("\n")
            else:
                file.write(",")


for Sx_name in np.sort(Sx_list):

    Sx = np.load(f"{PATH_SENSITIVE}/{Sx_name}", allow_pickle=True)
    match = re.search(r'p([\d\.]+)\.npy', Sx_name)
    sensitive_p = match.group(1)
    groups_value, groups_count = np.unique(Sx, return_counts=True)

    van_rows = np.load(PATH_GROUPS_VANILLA + f"/row_labels_p{sensitive_p}.npy")
    van_cols = np.load(PATH_GROUPS_VANILLA + f"/col_labels_p{sensitive_p}.npy")

    for fair_majority in fair_majority_range:

        for fair_minority in fair_minority_range:

            if groups_count[0] >= groups_count[1]:
                fair_params = np.array([fair_majority, fair_minority])
            else:
                fair_params = np.array([fair_minority, fair_majority])

            for run in run_range:
                print(f"majority = {fair_majority}, minority = {fair_minority}, run = {run}")

                start_time = 0
                end_time = 0

                start_time = time.time()

                #model = FairCoclusRows(initialization=INIT, verbose=True, k=10, l=10)
                if NUM_CLUSTERS < 10:
                    model = FairCoclus(initialization=INIT, verbose=True, k=10, l=10)
                else:
                    model = FairCoclus(initialization=INIT, verbose=True, k=20, l=20)

                model.fit(V=V, Sx=Sx, fair_row_parameters=fair_params)

                end_time = time.time()
                execution_time = end_time - start_time

                predict_rows = model.row_labels_
                predict_cols = model.column_labels_

                tau_x = model.tau_x[-1]
                tau_y = model.tau_y[-1]

                # NMI, AMI, ARI compared to true labels
                NMI_rows = normalized_mutual_info_score(true_rows, predict_rows)
                AMI_rows = adjusted_mutual_info_score(true_rows, predict_rows)
                ARI_rows = adjusted_rand_score(true_rows, predict_rows)

                NMI_cols = normalized_mutual_info_score(true_cols, predict_cols)
                AMI_cols = adjusted_mutual_info_score(true_cols, predict_cols)
                ARI_cols = adjusted_rand_score(true_cols, predict_cols)

                # NMI, AMI, ARI compared to vanilla predictions
                NMI_van_rows = normalized_mutual_info_score(van_rows, predict_rows)
                AMI_van_rows = adjusted_mutual_info_score(van_rows, predict_rows)
                ARI_van_rows = adjusted_rand_score(van_rows, predict_rows)

                NMI_van_cols = normalized_mutual_info_score(van_cols, predict_cols)
                AMI_van_cols = adjusted_mutual_info_score(van_cols, predict_cols)
                ARI_van_cols = adjusted_rand_score(van_cols, predict_cols)

                balance_bera = balance_gen(Sx, predict_rows)
                balance_chier = balance_chierichetti(Sx, predict_rows)

                predict_rows_np = np.array(predict_rows)
                K = len(np.unique(predict_rows_np))
                fair_error = KL_fairness_error(predict_rows_np, K, Sx)

                predict_cols_np = np.array(predict_cols)
                L = len(np.unique(predict_cols_np))

                num_iter = model._actual_n_iterations

                with open(PATH_RESULTS_FAIR + FILENAME_RESULTS, "a") as file:
                    file.write(f"{NUM_GROUPS},"
                               f"{sensitive_p},"
                               f"{fair_majority},"
                               f"{fair_minority},"
                               f"{run},"
                               f"{num_iter},"
                               f"{K},"
                               f"{L},"
                               f"{tau_x},"
                               f"{tau_y},"
                               f"{NMI_rows},"
                               f"{AMI_rows},"
                               f"{ARI_rows},"
                               f"{NMI_cols},"
                               f"{AMI_cols},"
                               f"{ARI_cols},"
                               f"{NMI_van_rows},"
                               f"{AMI_van_rows},"
                               f"{ARI_van_rows},"
                               f"{NMI_van_cols},"
                               f"{AMI_van_cols},"
                               f"{ARI_van_cols},"
                               f"{balance_chier},"
                               f"{balance_bera},"
                               f"{fair_error},"
                               f"{execution_time}\n"
                               )

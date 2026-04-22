import time
import re

from tauCC.src.fairness_metrics import balance_gen

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from config_synthetic_rows_cols import *

if ALGO_VERSION == "taucc_fair":
    from tauCC.src.taucc.taucc_fair_rows_cols import FairCoclus
elif ALGO_VERSION == "taucc_fair_max":
    from tauCC.src.taucc.taucc_fair_rows_cols_max import FairCoclus

FILENAME_RESULTS = f"/results_runs_groups{NUM_GROUPS}.csv"

results = {
    "num_groups": [],
    "sensitive_px": [],
    "sensitive_py": [],
    "row_fair_majority": [],
    "row_fair_minority": [],
    "col_fair_majority": [],
    "col_fair_minority": [],
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
    "balance_bera_rows": [],
    "balance_bera_cols": [],
    "time": []
}

results_keys = results.keys()
num_keys = len(results_keys)

van_rows = np.load(PATH_GROUPS_VANILLA + f"/best_row_labels.npy")
van_cols = np.load(PATH_GROUPS_VANILLA + f"/best_col_labels.npy")

if not os.path.exists(PATH_RESULTS_FAIR + FILENAME_RESULTS):
    with open(PATH_RESULTS_FAIR + FILENAME_RESULTS, "a") as file:
        for idx, key in enumerate(results_keys):
            file.write(f"{key}")
            if idx == num_keys - 1:
                file.write("\n")
            else:
                file.write(",")

for Sx_name in np.sort(Sx_list):
    Sx = np.load(f"{PATH_SENSITIVE_X}/{Sx_name}", allow_pickle=True)
    match_x = re.search(r'sx_p([\d\.]+)\.npy', Sx_name)
    sensitive_px = match_x.group(1)
    groups_value_x, groups_count_x = np.unique(Sx, return_counts=True)

    for Sy_name in np.sort(Sy_list):
        Sy = np.load(f"{PATH_SENSITIVE_Y}/{Sy_name}", allow_pickle=True)
        match_y = re.search(r'sy_p([\d\.]+)\.npy', Sy_name)
        sensitive_py = match_y.group(1)
        groups_value_y, groups_count_y = np.unique(Sy, return_counts=True)

        for row_fair_majority in row_fair_majority_range:
            for row_fair_minority in row_fair_minority_range:

                # Assign row fair params respecting majority/minority order
                if groups_count_x[0] >= groups_count_x[1]:
                    row_fair_params = np.array([row_fair_majority, row_fair_minority])
                else:
                    row_fair_params = np.array([row_fair_minority, row_fair_majority])

                for col_fair_majority in col_fair_majority_range:
                    for col_fair_minority in col_fair_minority_range:

                        # Assign col fair params respecting majority/minority order
                        if groups_count_y[0] >= groups_count_y[1]:
                            col_fair_params = np.array([col_fair_majority, col_fair_minority])
                        else:
                            col_fair_params = np.array([col_fair_minority, col_fair_majority])

                        for run in run_range:
                            print(f"px={sensitive_px}, py={sensitive_py} | "
                                  f"row_maj={row_fair_majority}, row_min={row_fair_minority} | "
                                  f"col_maj={col_fair_majority}, col_min={col_fair_minority} | "
                                  f"run={run}")

                            start_time = time.time()

                            if NUM_CLUSTERS < 10:
                                model = FairCoclus(initialization=INIT, verbose=True, k=10, l=10)
                            else:
                                model = FairCoclus(initialization=INIT, verbose=True, k=20, l=20)

                            model.fit(
                                V=V,
                                Sx=Sx,
                                Sy=Sy,
                                fair_row_parameters=row_fair_params,
                                fair_col_parameters=col_fair_params
                            )

                            end_time = time.time()
                            execution_time = end_time - start_time

                            predict_rows = model.row_labels_
                            predict_cols = model.column_labels_

                            tau_x = model.tau_x[-1]
                            tau_y = model.tau_y[-1]

                            # Clustering quality vs true labels
                            NMI_rows = normalized_mutual_info_score(true_rows, predict_rows)
                            AMI_rows = adjusted_mutual_info_score(true_rows, predict_rows)
                            ARI_rows = adjusted_rand_score(true_rows, predict_rows)

                            NMI_cols = normalized_mutual_info_score(true_cols, predict_cols)
                            AMI_cols = adjusted_mutual_info_score(true_cols, predict_cols)
                            ARI_cols = adjusted_rand_score(true_cols, predict_cols)

                            # Clustering quality vs vanilla baseline
                            NMI_van_rows = normalized_mutual_info_score(van_rows, predict_rows)
                            AMI_van_rows = adjusted_mutual_info_score(van_rows, predict_rows)
                            ARI_van_rows = adjusted_rand_score(van_rows, predict_rows)

                            NMI_van_cols = normalized_mutual_info_score(van_cols, predict_cols)
                            AMI_van_cols = adjusted_mutual_info_score(van_cols, predict_cols)
                            ARI_van_cols = adjusted_rand_score(van_cols, predict_cols)

                            # Row fairness metrics
                            predict_rows_np = np.array(predict_rows)
                            K = len(np.unique(predict_rows_np))
                            balance_bera_rows = balance_gen(Sx, predict_rows_np)

                            # Col fairness metrics
                            predict_cols_np = np.array(predict_cols)
                            L = len(np.unique(predict_cols_np))
                            balance_bera_cols = balance_gen(Sy, predict_cols_np)

                            num_iter = model._actual_n_iterations

                            with open(PATH_RESULTS_FAIR + FILENAME_RESULTS, "a") as file:
                                file.write(f"{NUM_GROUPS},"
                                           f"{sensitive_px},"
                                           f"{sensitive_py},"
                                           f"{row_fair_majority},"
                                           f"{row_fair_minority},"
                                           f"{col_fair_majority},"
                                           f"{col_fair_minority},"
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
                                           f"{balance_bera_rows},"
                                           f"{balance_bera_cols},"
                                           f"{execution_time}\n"
                                           )
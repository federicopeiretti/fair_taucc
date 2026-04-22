import re
import time

from tauCC.src.taucc.taucc import CoClust
from tauCC.src.fairness_metrics import balance_gen

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from config_synthetic_rows_cols import *

FILENAME_RESULTS = f"/results_runs_groups{NUM_GROUPS}.csv"

root = os.path.dirname(os.path.realpath(__file__))
print(f"root: {root}")

results = {
    "num_groups": [],
    "sensitive_px": [],
    "sensitive_py": [],
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
    "balance_bera_rows": [],
    "balance_bera_cols": [],
    "time": []
}

results_keys = results.keys()
num_keys = len(results_keys)

if not os.path.exists(PATH_RESULTS_VANILLA + FILENAME_RESULTS):
    with open(PATH_RESULTS_VANILLA + FILENAME_RESULTS, "a") as file:
        for idx, key in enumerate(results_keys):
            file.write(f"{key}")
            if idx == num_keys - 1:
                file.write("\n")
            else:
                file.write(";")


# ── Step 1: run the vanilla algorithm (independent of sensitive features) ──────
max_tau_x = 0
max_tau_y = 0
best_predict_rows = None
best_predict_cols = None

# Store per-run results to avoid re-running when computing fairness metrics
run_results = []   # list of dicts: {run, predict_rows, predict_cols, tau_x, tau_y, ...}

for run in run_range:

    start_time = time.time()

    if NUM_CLUSTERS < 10:
        model = CoClust(initialization="random", verbose=True, k=10, l=10)
    else:
        model = CoClust(initialization="random", verbose=True, k=20, l=20)

    model.fit(V)

    execution_time = time.time() - start_time

    predict_rows = np.array(model.row_labels_)
    predict_cols = np.array(model.column_labels_)

    tau_x = model.tau_x[-1]
    tau_y = model.tau_y[-1]
    num_iter = model._actual_n_iterations
    K = len(np.unique(predict_rows))
    L = len(np.unique(predict_cols))

    # Clustering quality vs true labels (independent of sensitive features)
    NMI_rows = normalized_mutual_info_score(true_rows, predict_rows)
    AMI_rows = adjusted_mutual_info_score(true_rows, predict_rows)
    ARI_rows = adjusted_rand_score(true_rows, predict_rows)

    NMI_cols = normalized_mutual_info_score(true_cols, predict_cols)
    AMI_cols = adjusted_mutual_info_score(true_cols, predict_cols)
    ARI_cols = adjusted_rand_score(true_cols, predict_cols)

    run_results.append({
        "run": run,
        "predict_rows": predict_rows,
        "predict_cols": predict_cols,
        "tau_x": tau_x,
        "tau_y": tau_y,
        "num_iter": num_iter,
        "K": K,
        "L": L,
        "NMI_rows": NMI_rows, "AMI_rows": AMI_rows, "ARI_rows": ARI_rows,
        "NMI_cols": NMI_cols, "AMI_cols": AMI_cols, "ARI_cols": ARI_cols,
        "time": execution_time,
    })

    # Track best solution in terms of tau_x and tau_y
    if tau_x > max_tau_x and tau_y > max_tau_y:
        max_tau_x = tau_x
        max_tau_y = tau_y
        best_predict_rows = predict_rows.copy()
        best_predict_cols = predict_cols.copy()

# Save best vanilla clustering (used as baseline in fair experiments)
np.save(PATH_GROUPS_VANILLA + f"/best_row_labels.npy", best_predict_rows)
np.save(PATH_GROUPS_VANILLA + f"/best_col_labels.npy", best_predict_cols)
print(f"Best vanilla solution saved (tau_x={max_tau_x:.4f}, tau_y={max_tau_y:.4f})")


# ── Step 2: compute fairness metrics for every (Sx, Sy) pair ──────────────────
for Sx_name in np.sort(Sx_list):
    Sx = np.load(f"{PATH_SENSITIVE_X}/{Sx_name}", allow_pickle=True)
    match_x = re.search(r'sx_p([\d\.]+)\.npy', Sx_name)
    sensitive_px = match_x.group(1)

    for Sy_name in np.sort(Sy_list):
        Sy = np.load(f"{PATH_SENSITIVE_Y}/{Sy_name}", allow_pickle=True)
        match_y = re.search(r'sy_p([\d\.]+)\.npy', Sy_name)
        sensitive_py = match_y.group(1)

        for r in run_results:
            predict_rows = r["predict_rows"]
            predict_cols = r["predict_cols"]
            K = r["K"]
            L = r["L"]

            # Fairness metrics
            balance_bera_rows  = balance_gen(Sx, predict_rows)
            balance_bera_cols  = balance_gen(Sy, predict_cols)

            with open(PATH_RESULTS_VANILLA + FILENAME_RESULTS, "a") as file:
                file.write(f"{NUM_GROUPS};"
                           f"{sensitive_px};"
                           f"{sensitive_py};"
                           f"{r['run']};"
                           f"{r['num_iter']};"
                           f"{K};"
                           f"{L};"
                           f"{r['tau_x']};"
                           f"{r['tau_y']};"
                           f"{r['NMI_rows']};"
                           f"{r['AMI_rows']};"
                           f"{r['ARI_rows']};"
                           f"{r['NMI_cols']};"
                           f"{r['AMI_cols']};"
                           f"{r['ARI_cols']};"
                           f"{balance_bera_rows};"
                           f"{balance_bera_cols};"
                           f"{r['time']}\n"
                           )
import os
import numpy as np
from tauCC.src.utils import create_path
import glob

INIT = "random"     # initialization
RUNS = 10           # fair runs

NUM_CLUSTERS = 3
NUM_GROUPS = 2

#ALGO_VERSION = "taucc_fair"    # fair_taucc_v1
ALGO_VERSION = "taucc_fair_max" # fair_taucc_v2

# *** DATASET - start ***
root = os.path.dirname(os.path.realpath(__file__))
print(f"root: {root}")

PATH_DATASET = f"{root}/datasets/synthetic/clus{NUM_CLUSTERS}_rc"
PATH_SENSITIVE_X = f"{PATH_DATASET}/groups{NUM_GROUPS}/rows"   # sensitive features for rows (sx_p*.npy)
PATH_SENSITIVE_Y = f"{PATH_DATASET}/groups{NUM_GROUPS}/cols"   # sensitive features for cols (sy_p*.npy)

PATH_RESULTS = f"{root}/results/synthetic"
create_path(PATH_RESULTS)
PATH_RESULTS += f"/clus{NUM_CLUSTERS}_rc"
create_path(PATH_RESULTS)

PATH_RESULTS_VANILLA = PATH_RESULTS + f"/taucc_vanilla"
create_path(PATH_RESULTS_VANILLA)
PATH_GROUPS_VANILLA = PATH_RESULTS_VANILLA + f"/groups{NUM_GROUPS}"
create_path(PATH_GROUPS_VANILLA)

PATH_RESULTS_FAIR = PATH_RESULTS + f"/{ALGO_VERSION}"
create_path(PATH_RESULTS_FAIR)

V = np.load(PATH_DATASET + f"/matrix.npy", allow_pickle=True)
true_rows = np.load(PATH_DATASET + f"/row_labels.npy", allow_pickle=True)
true_cols = np.load(PATH_DATASET + f"/col_labels.npy", allow_pickle=True)

# Row sensitive files: sx_p*.npy
Sx_files = glob.glob(os.path.join(PATH_SENSITIVE_X, "sx_p*.npy"))
Sx_list = np.sort([os.path.basename(f) for f in Sx_files])

# Col sensitive files: sy_p*.npy
Sy_files = glob.glob(os.path.join(PATH_SENSITIVE_Y, "sy_p*.npy"))
Sy_list = np.sort([os.path.basename(f) for f in Sy_files])

print(f"Dataset: synthetic")
print(f"init: {INIT}")
print(f"shape of V: {V.shape}")
print(f"Sx files: {Sx_list}")
print(f"Sy files: {Sy_list}", flush=True)
# *** DATASET - end ***

fair_param_range = np.arange(0.0, 1.25, 0.25)
#Sx_list = Sx_list[[0, 2, 4, 5]]
#Sy_list = Sy_list[[0, 2, 4, 5]]

Sx_list = Sx_list[[4]]
Sy_list = Sy_list[[4]]

#fair_param_range = np.array([1.0])
#Sx_list = Sx_list[[5]]
#Sy_list = Sy_list[[5]]

# Row fairness parameter ranges
row_fair_majority_range = np.round(fair_param_range, 3)
row_fair_minority_range = np.round(fair_param_range, 3)

# Col fairness parameter ranges
col_fair_majority_range = np.round(fair_param_range, 3)
col_fair_minority_range = np.round(fair_param_range, 3)

run_range = np.arange(0, RUNS)

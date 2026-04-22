import os
import numpy as np
from tauCC.src.utils import create_path
import glob

INIT = "random"     # initialization
RUNS = 10           # fair runs

NUM_CLUSTERS = 5
NUM_GROUPS = 3

#ALGO_VERSION = "taucc_fair"    # fair_taucc_v1
ALGO_VERSION = "taucc_fair_max" # fair_taucc_v2

# *** DATASET - start ***
root = os.path.dirname(os.path.realpath(__file__))
print(f"root: {root}")

PATH_DATASET = f"{root}/datasets/synthetic/clus{NUM_CLUSTERS}"
PATH_SENSITIVE = f"{PATH_DATASET}/groups{NUM_GROUPS}"

PATH_RESULTS = f"{root}/results/synthetic"
create_path(PATH_RESULTS)
PATH_RESULTS += f"/clus{NUM_CLUSTERS}"
create_path(PATH_RESULTS)

PATH_RESULTS_VANILLA = PATH_RESULTS + f"/taucc_vanilla"
create_path(PATH_RESULTS_VANILLA)
PATH_GROUPS_VANILLA = PATH_RESULTS_VANILLA + f"/groups{NUM_GROUPS}"
create_path(PATH_GROUPS_VANILLA)

PATH_RESULTS_FAIR = PATH_RESULTS + f"/{ALGO_VERSION}"
create_path(PATH_RESULTS_FAIR)
#PATH_GROUPS_FAIR = PATH_RESULTS_FAIR + f"/groups{NUM_GROUPS}"
#create_path(PATH_GROUPS_FAIR)

V = np.load(PATH_DATASET + f"/matrix.npy", allow_pickle=True)
true_rows = np.load(PATH_DATASET + f"/row_labels.npy", allow_pickle=True)
true_cols = np.load(PATH_DATASET + f"/col_labels.npy", allow_pickle=True)

Sx_files = glob.glob(os.path.join(PATH_SENSITIVE, "*.npy"))
Sx_list = np.sort([os.path.basename(file) for file in Sx_files])

print(f"Dataset: synthetic")
print(f"init {INIT}")
print(f"shape of V: {V.shape}", flush=True)
# *** DATASET - end ***

#if NUM_GROUPS == 2 or NUM_CLUSTERS <= 5:
#    fair_param_range = np.arange(0.0, 1.10, 0.10)
#else:
fair_param_range = np.arange(0.0, 1.25, 0.25)
fair_minority_range2 = np.round(fair_param_range, 3)
Sx_list = Sx_list[[0,3,5,8,10]]

fair_majority_range = np.round(fair_param_range, 3)
fair_minority_range = np.round(fair_param_range, 3)

run_range = np.arange(0, RUNS)

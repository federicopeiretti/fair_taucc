"""
    Configuration file
"""


import os
import numpy as np
import pandas as pd
from tauCC.src.utils import create_path

INIT = "random"     # initialization
#RUNS = 30          # vanilla runs
RUNS = 10           # fair runs


DATASET = "movielens-1m"
SENSITIVE = "gender"
#SENSITIVE = "age"
TRUE_LABEL = "genres"
TRUE_LABEL_DIM = "cols"


"""
DATASET = "amazon"
SENSITIVE = "gender"
TRUE_LABEL = "preferred_words_by_category"
TRUE_LABEL_DIM = "cols"
"""

"""
DATASET = "yelp"
SENSITIVE = "gender"
TRUE_LABEL = "restaurant_type"
TRUE_LABEL_DIM = "cols"
"""

"""
DATASET = "lfw"
SENSITIVE = "gender"
TRUE_LABEL = "person_ids"
TRUE_LABEL_DIM = "rows"
"""

"""
DATASET = "rfw"
SENSITIVE = "race"
TRUE_LABEL = "person_ids"
TRUE_LABEL_DIM = "rows"
"""



# *** DATASET - start ***
root = os.path.dirname(os.path.realpath(__file__))
# root = root.replace("\\", "/")
print(f"root: {root}")

if "movielens" in DATASET:
    PATH_DATASET = f"{root}/datasets/movielens/{DATASET}"
else:
    PATH_DATASET = f"{root}/datasets/{DATASET}"

V = np.load(PATH_DATASET + f"/matrix.npy", allow_pickle=True).astype(int)
Sx = np.load(PATH_DATASET + f"/{SENSITIVE}.npy", allow_pickle=True).astype(int)

if TRUE_LABEL != " ":

    true_labels = np.load(PATH_DATASET + f"/{TRUE_LABEL}.npy", allow_pickle=True).astype(int)

    if true_labels.ndim != 1:
        true_labels = true_labels.reshape(-1)


groups, count_groups = np.unique(Sx, return_counts=True)

print(f"Dataset: {DATASET}")
print(f"init {INIT}")
print(f"shape of V: {V.shape}", flush=True)
print(f"shape of Sx ({SENSITIVE}): {Sx.shape}", flush=True)
print(f"protected groups: {groups}", flush=True)
print(f"count protected groups: {count_groups}", flush=True)
# *** DATASET - end ***

fair_majority_range = np.round(np.arange(0.0, 1.10, 0.10), 3)
fair_minority_range = np.round(np.arange(0.0, 1.10, 0.10), 3)
#fair_minority_range2 = np.round(np.arange(0.0, 1.10, 0.10), 3)

run_range = np.arange(0, RUNS)

# *** PATH - start ***
PATH_RESULTS = f"{root}/results/{DATASET}"
create_path(PATH_RESULTS)

PATH_RESULTS += f"/{SENSITIVE}"
create_path(PATH_RESULTS)

PATH_RESULTS_VANILLA = PATH_RESULTS + f"/taucc_vanilla"
PATH_RESULTS_VANILLA_INIT = PATH_RESULTS_VANILLA + f"/init_{INIT}"

PATH_RESULTS_FAIR = PATH_RESULTS + f"/taucc_fair"
PATH_RESULTS_FAIR_INIT = PATH_RESULTS_FAIR + f"/init_{INIT}"

create_path(PATH_RESULTS_VANILLA)
create_path(PATH_RESULTS_VANILLA_INIT)
create_path(PATH_RESULTS_FAIR)
create_path(PATH_RESULTS_FAIR_INIT)

# *** PATH - end ***

if os.path.exists(PATH_RESULTS_VANILLA_INIT + "/best_run.csv"):
    df_best_run = pd.read_csv(PATH_RESULTS_VANILLA_INIT + "/best_run.csv")
    best_run = df_best_run["run"].values[0]
    vanilla_rows = np.load(PATH_RESULTS_VANILLA_INIT + f"/run{best_run}/data/row_assignment.npy")
    vanilla_cols = np.load(PATH_RESULTS_VANILLA_INIT + f"/run{best_run}/data/col_assignment.npy")


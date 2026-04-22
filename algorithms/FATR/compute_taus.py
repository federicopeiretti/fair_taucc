from __future__ import division
import numpy as np


def relabel_consecutive(arr):
    unique_values = np.unique(arr)
    if np.all(np.arange(0, len(unique_values)) == unique_values):
        return arr
    else:
        print("-- Relabel Consecutive --")
        arr = np.asarray(arr)
        unique_vals, inverse = np.unique(arr, return_inverse=True)
        mapping = np.arange(len(unique_vals))
        if 0 in unique_vals:
            zero_idx = np.where(unique_vals == 0)[0][0]
            mapping[zero_idx + 1:] = np.arange(1, len(unique_vals))
        else:
            mapping = np.arange(0, len(unique_vals))
        return mapping[inverse]


def get_row_col_incidence(clus):
    num_clus = len(np.unique(clus))
    num_items = len(clus)
    incidence = np.zeros((num_items, num_clus))
    #incidence[np.arange(0,num_items, dtype=int), clus.astype(int)] = 1
    incidence[np.arange(0, num_items, dtype=int), clus] = 1
    return incidence


def update_dataset(dataset, row_incidence, col_incidence, dimension):
    if dimension == 0:
        new_t = np.dot(dataset, col_incidence)
    else:
        new_t = np.dot(row_incidence.T, dataset)
    return new_t


def contingency_matrix(dataset, dimension, row_clus, col_clus):
    row_incidence = get_row_col_incidence(row_clus)
    col_incidence = get_row_col_incidence(col_clus)
    dataset = update_dataset(dataset, row_incidence, col_incidence, dimension)
    if dimension == 0:
        new_t = np.dot(row_incidence.T, dataset)
    else:
        new_t = np.dot(dataset, col_incidence)
    return dataset, new_t


def compute_taus(dataset, dimension, row_clus, col_clus):
    row_clus = relabel_consecutive(row_clus)
    col_clus = relabel_consecutive(col_clus)
    _, T = contingency_matrix(dataset, dimension, row_clus, col_clus)
    tot_per_x = np.sum(T, 1)
    tot_per_y = np.sum(T, 0)
    t_square = np.power(T, 2)
    a_x = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 0), tot_per_y)))
    b_x = np.sum(np.power(tot_per_x, 2))
    a_y = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 1), tot_per_x)))
    b_y = np.sum(np.power(tot_per_y, 2))
    tau_x = np.nan_to_num(np.true_divide(a_x - b_x, 1 - b_x))
    tau_y = np.nan_to_num(np.true_divide(a_y - b_y, 1 - b_y))
    return tau_x, tau_y, (a_x - b_x), (a_y - b_y)


"""
dataset = "movielens-1m"
sensitive = "age"

V = np.load(f"datasets/movielens/{dataset}/matrix.npy")
V = V/np.sum(V)

#algorithm = "frisch_lbm_baseline"
algorithm = "frisch_lbm_fair"

if algorithm == "frisch_lbm_baseline":
    path = f"algorithms/C-Fairness-RecSys/reproducibility_study/Frisch_et_al/results/{dataset}/{sensitive}/lbm_baseline"
elif algorithm == "frisch_lbm_fair":
    path = f"algorithms/C-Fairness-RecSys/reproducibility_study/Frisch_et_al/results/{dataset}/{sensitive}/lbm_fair"

df = pd.read_csv(path + "/results.csv", sep=";")
df.sort_values("run", inplace=True)
taus_x = []
taus_y = []

for run in df["run"]:
    row_clus = np.load(f"{path}/run_{run}_row_clustering.npy")
    col_clus = np.load(f"{path}/run_{run}_col_clustering.npy")
    tau_x, tau_y, _, _ = compute_taus(V, 0, row_clus, col_clus)
    taus_x.append(tau_x)
    taus_y.append(tau_y)

df["tau_x"] = taus_x
df["tau_y"] = taus_y
df.to_csv(path + "/results_with_taus.csv", index=False)

print(f"ALGORITHM: {algorithm}")
print(f"DATASET: {dataset}")
print("tau x")
print("mean: ", np.mean(taus_x))
print("std: ", np.std(taus_x))
print("var: ", np.var(taus_x))
print()
print("tau y")
print("mean: ", np.mean(taus_y))
print("std: ", np.std(taus_y))
print("var: ", np.var(taus_y))
"""
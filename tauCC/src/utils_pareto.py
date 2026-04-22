import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import skcriteria as skc
from skcriteria.preprocessing import invert_objectives, scalers
from skcriteria.agg import similarity
from skcriteria.pipeline import mkpipe


def find_pareto_front_3d(tau_x, tau_y, balance):
    """Find the Pareto frontier for 3 objectives to be maximized"""
    points = np.column_stack([tau_x, tau_y, balance])
    pareto_indices = []

    for i, point in enumerate(points):
        is_dominated = False
        for j, other_point in enumerate(points):
            if i != j:
                # Un punto domina un altro se è >= in tutti gli obiettivi
                # e > in almeno uno (per massimizzazione)
                if (other_point >= point).all() and (other_point > point).any():
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_indices.append(i)

    return pareto_indices


def plot_pareto_3d(tau_x, tau_y, balance, pareto_indices, path, text=None):
    fig = plt.figure(figsize=(12, 5))

    # Plot 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(tau_x, tau_y, balance, alpha=0.7, s=50)
    ax1.scatter([tau_x[i] for i in pareto_indices],
                [tau_y[i] for i in pareto_indices],
                [balance[i] for i in pareto_indices],
                color='red', s=100, label='Pareto frontier')

    for value in pareto_indices:
        ax1.text(x=float(tau_x[value]), y=float(tau_y[value]), z=float(balance[value]), s=value)

    ax1.set_xlabel('tau_x')
    ax1.set_ylabel('tau_y')
    ax1.set_zlabel('balance')
    ax1.legend()
    ax1.set_title('Pareto frontier (3D)')

    # Plot 2D proiezioni
    ax2 = fig.add_subplot(122)
    ax2.scatter(tau_x, tau_y, alpha=0.7, s=50)
    ax2.scatter([tau_x[i] for i in pareto_indices],
                [tau_y[i] for i in pareto_indices],
                color='red', s=100, label='Pareto frontier')

    for value in pareto_indices:
        ax2.annotate(value, (float(tau_x[value]), float(tau_y[value])))

    ax2.set_xlabel('tau_x')
    ax2.set_ylabel('tau_y')
    ax2.legend()
    ax2.set_title('Projection tau_x vs tau_y')

    plt.tight_layout()
    # plt.show()
    # if text is not None:
    #    plt.savefig(path + f"/{text}.png", bbox_inches="tight")
    # else:
    #    plt.savefig(path + f"/pareto.png", bbox_inches="tight")


def topsis_analysis(tau_x, tau_y, balance):
    """TOPSIS for ranking"""
    try:
        # Creation of the decision matrix
        data = np.column_stack([tau_x, tau_y, balance])
        dm = skc.mkdm(matrix=data,
                      objectives=[max, max, max],
                      alternatives=[f"Run_{i}" for i in range(len(tau_x))])

        pipe = mkpipe(
            invert_objectives.NegateMinimize(),
            scalers.VectorScaler(target="matrix"),
            scalers.SumScaler(target="weights"),
            similarity.TOPSIS(),
        )

        result = pipe.evaluate(dm)

        df_results = pd.DataFrame({
            'id_row': [i for i in range(len(tau_x))],
            'tau_x': tau_x,
            'tau_y': tau_y,
            'balance': balance,
            'TOPSIS_score': result.e_.similarity,
            'rank': result.rank_
        }).sort_values('rank')

        best_run_idx = df_results.loc[df_results["rank"].idxmin()]["id_row"].astype(int)
        return df_results, best_run_idx

    except ImportError:
        print("Install scikit-criteria: pip install scikit-criteria")
        return None, None

def best_run_vanilla(root, dataset, sensitive, init="random"):
    path = f"{root}/results/{dataset}/{sensitive}/taucc_vanilla/init_{init}"
    df = pd.read_csv(path + "/results_runs.csv", sep=";")
    best_run_row = df.iloc[np.argmax(df["tau_x"])]
    best_run = best_run_row["run"].astype(int)
    print(f"Dataset: {dataset} ({sensitive})")
    print(f"Best run: {best_run}")
    # pd.DataFrame(best_run_row).to_csv(path + "/best_run.csv", index=False)


def best_parameters_fair(root, dataset, sensitive, algorithm="taucc_fair_max", init="random"):
    print(f"dataset: {dataset}")
    print(f"sensitive: {sensitive}")

    path = f"{root}/results/{dataset}/{sensitive}/{algorithm}/init_{init}"
    df = pd.read_csv(path + "/aggregated.csv")

    tau_x_scores = df["tau_x_mean"].to_numpy()
    tau_y_scores = df["tau_y_mean"].to_numpy()
    balance_scores = df["balance_bera_mean"].to_numpy()

    """
    # Pareto
    pareto_indices = find_pareto_front_3d(tau_x_scores, tau_y_scores, balance_scores)
    plot_pareto_3d(tau_x_scores, tau_y_scores, balance_scores, pareto_indices, path)
    df_pareto = df.iloc[pareto_indices][
        ["fair_majority", "fair_minority", "tau_x_mean", "tau_y_mean", "balance_bera_mean", "balance_bera_std"]]
    #df_pareto.to_csv(path + "/pareto.csv", index=False)
    """

    # TOPSIS: topsis_results (ranking), best_topsis (best index)
    topsis_results, best_topsis = topsis_analysis(tau_x_scores, tau_y_scores, balance_scores)
    topsis_results.to_csv(path + "/topsis_rank.csv", index=False)
    result = df.iloc[best_topsis].to_frame().T.reset_index(drop=True)
    result.to_csv(path + "/topsis_best_param.csv", index=False)

    """
    #print(f"Pareto frontiers: {pareto_indices}")
    print(f"Topsis results: {best_topsis}")
    if groups == 2:
        print(df.iloc[best_topsis][["fair_majority", "fair_minority", "tau_x_mean", "tau_y_mean", "balance_bera_mean"]])
    else:
        print(df.iloc[best_topsis][
                  ["fair_majority", "fair_minority", "fair_minority2", "tau_x_mean", "tau_y_mean", "balance_bera_mean"]])
    print()
    """


def best_run_fair(root, dataset, sensitive, groups, algorithm="taucc_fair_max", init="random"):
    path = f"{root}/results/{dataset}/{sensitive}/{algorithm}/init_{init}"
    df = pd.read_csv(path + "/best_run.csv", sep=";")
    tau_x_scores = df["tau_x"].to_numpy()
    tau_y_scores = df["tau_y"].to_numpy()
    balance_scores = df["balance_bera"].to_numpy()

    """
    # Pareto
    pareto_indices = find_pareto_front_3d(tau_x_scores, tau_y_scores, balance_scores)
    plot_pareto_3d(tau_x_scores, tau_y_scores, balance_scores, pareto_indices, path, text="best_run_pareto")
    df_pareto = df.iloc[pareto_indices][
        ["fair_majority", "fair_minority", "tau_x", "tau_y", "balance_bera"]]
    #df_pareto.to_csv(path + "/best_run_pareto.csv", index=False)
    """

    # TOPSIS
    topsis_results, best_topsis = topsis_analysis(tau_x_scores, tau_y_scores, balance_scores)
    topsis_results.to_csv(path + "/topsis_best_run.csv", index=False)

    # print(f"Pareto frontiers: {pareto_indices}")
    print(f"Topsis results: {best_topsis}")
    if groups == 2:
        print(df.iloc[best_topsis][["fair_majority", "fair_minority", "tau_x_mean", "tau_y_mean", "balance_bera_mean"]])
    else:
        print(df.iloc[best_topsis][
                  ["fair_majority", "fair_minority", "fair_minority2", "tau_x_mean", "tau_y_mean",
                   "balance_bera_mean"]])
    print()


def best_parameters_fair_synth(root, clusters, groups, algorithm, rc=False):
    print(f"dataset: synthetic")
    print(f"clusters: {clusters}, groups: {groups}")

    final_result = pd.DataFrame()

    if not rc:
        path = f"{root}/results/synthetic/clus{clusters}/{algorithm}"
        df = pd.read_csv(path + f"/aggregated_groups{groups}.csv")
        sensitive_p = df["sensitive_p"].unique()

        for p in sensitive_p:
            df_filter = df.query(f"sensitive_p == {p}")
            tau_x_scores = df_filter["tau_x_mean"].to_numpy()
            tau_y_scores = df_filter["tau_y_mean"].to_numpy()
            balance_scores = df_filter["balance_bera_mean"].to_numpy()

            _, best_topsis = topsis_analysis(tau_x_scores, tau_y_scores, balance_scores)
            #topsis_results.to_csv(path + "/topsis_rank.csv", index=False)
            result = df_filter.iloc[best_topsis].to_frame().T.reset_index(drop=True)
            final_result = pd.concat([final_result, result], ignore_index=True)

        final_result.to_csv(path + f"/topsis_best_param_groups{groups}.csv", index=False)

    else:
        path = f"{root}/results/synthetic/clus{clusters}_rc/{algorithm}"
        df = pd.read_csv(path + f"/aggregated_groups{groups}.csv")
        sensitive_px = df["sensitive_px"].unique()
        sensitive_py = df["sensitive_py"].unique()

        for px in sensitive_px:
            for py in sensitive_py:
                df_filter = df.query(f"sensitive_px == {px} and sensitive_py == {py}")
                tau_x_scores = df_filter["tau_x_mean"].to_numpy()
                tau_y_scores = df_filter["tau_y_mean"].to_numpy()
                balance_scores = df_filter["balance_bera_mean"].to_numpy()

                topsis_results, best_topsis = topsis_analysis(tau_x_scores, tau_y_scores, balance_scores)
                topsis_results.to_csv(path + "/topsis_rank.csv", index=False)
                result = df_filter.iloc[best_topsis].to_frame().T.reset_index(drop=True)
                final_result = pd.concat([final_result, result], ignore_index=True)

        final_result.to_csv(path + f"/topsis_best_param_groups{groups}_rc.csv", index=False)
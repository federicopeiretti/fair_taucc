from lbm_ordinal import LbmOrdinal
from lbm_ordinal2 import LbmOrdinal as LbmOrdinalBaseline
import pickle
import time
import torch
import numpy as np
import time
import warnings
import argparse
import glob
import os
import errno
from sklearn import metrics
from sklearn.metrics import ndcg_score

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        #print("Folder %s created!" % path)
    #else:
    #print("Folder %s already exists" % path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--block", default=0, type=int)
#parser.add_argument("--nq", default=25, type=int)
#parser.add_argument("--nl", default=25, type=int)
parser.add_argument("--nq", default=10, type=int)
parser.add_argument("--nl", default=10, type=int)

args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")
warnings.filterwarnings("ignore")

###################################################################
###################### ADDED/MODIFIED CODE ########################

#sensitive_attribute = "user_age"
#dataset = "filtered(20)_lastfm-1K"

#sensitive_attribute = "user_gender"
#dataset = "movielens_1m"

sensitive_attribute = "gender"
dataset = "lfw"

data_file = f"{dataset}_blocks_topk_{sensitive_attribute}"
print(data_file)

root = os.path.abspath(os.path.dirname(__file__))
print(root)

data = pickle.load(open(f"{root}/data/{dataset}/{data_file}.pkl", "rb"))
block = args.block

X_train = data["blocks"][block]["X_train"]

# Last.FM ratings are float. Co-clustering needs int
if not isinstance(X_train.getrow(0).getcol(0).todense()[0, 0], np.integer):
    X_train = round(X_train).astype(np.int64)

X_train = np.array(X_train.todense())

"""
X_test = data["blocks"][block]["X_test"]

if not isinstance(X_test.getrow(0).getcol(0).todense()[0, 0], np.integer):
    X_test = round(X_test).astype(np.int64)

X_test = np.array(X_test.todense())
"""
###################################################################
###################################################################

X = X_train
#X = np.add(X_train, X_test)
n1, n2 = X.shape
nq, nl = args.nq, args.nl
nk = 5

females = (
    data["data_users"].userid[data["data_users"].gender == "F"].to_numpy()
)
males = data["data_users"].userid[data["data_users"].gender == "M"].to_numpy()
groups = (females, males)

if X_train.shape == X.shape:
    covariates = np.array(np.zeros_like(X_train))
    covariates[males] = 1
    covariates[females] = -1
    covariates = covariates.reshape(X_train.shape[0], X_train.shape[1], 1)
    covariates = covariates.astype(float)
else:
    raise ValueError("X_train.shape != X.shape")

# test_data = list(  # np.array(
#     [X_test[i, X_test[i, ].nonzero()].flatten().tolist() for i in range(n1)]
# )

######################## SET TO NONE TO COMPUTE BASELINE ##########################
# covariates = None
###################################################################################

directory = root + f"/results/{dataset}/"

create_path(directory)

if covariates is not None:
    directory += "lbm_fair/"
else:
    directory += "lbm_baseline/"

if not os.path.exists(directory):
    try:
        os.makedirs(directory)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

to_save = []
if covariates is not None:
    run_files = [f for f in os.listdir(directory) if sensitive_attribute in f and "run" in f]
    start_run = max(map(lambda x: int(x.split('_')[3]), run_files)) + 1 if run_files else 0
else:
    start_run = 0
print(f"{sensitive_attribute} start_run: {start_run}")

if not os.path.exists(directory + "time.csv"):
    with open(directory + "time.csv", "a") as file:
        file.write(f"run,time\n")

RUNS = 10

for run in range(start_run, RUNS):  # 25 replicates of the experiment as stated in the paper
    print(sensitive_attribute)
    try:
        if covariates is not None:
            print("LbmOrdinal")
            model = LbmOrdinal(device=device)
        else:
            print("LbmOrdinal - Baseline")
            model = LbmOrdinalBaseline(device=device)

        def callback(model, epoch=0):
            chi2 = torch.stack([model.tau_1[gr].sum(0) for gr in groups])
            chi2_stat = (
                (
                        chi2
                    - (
                        chi2.sum(0).reshape(1, nq)
                        * chi2.sum(1).reshape(len(groups), 1)
                    )
                    / chi2.sum()
                )
                ** 2
                / (
                    (
                        chi2.sum(0).reshape(1, nq)
                        * chi2.sum(1).reshape(len(groups), 1)
                    )
                    / chi2.sum()
                )
            ).sum()

            pred_cov = (
                (
                    model.tau_1 @ model.pi @ model.tau_2.T
                    + model.eta_row
                    + model.eta_col
                    + 1
                )
                .detach()
                .cpu()
                .numpy()
            )
            # pred_test_cov = list(  # np.array(
            #     [
            #         pred_cov[i, X_test[i,].nonzero()].flatten().tolist()
            #         for i in range(n1)
            #     ]
            # )

            # _test_data = test_data  # np.array(test_data[:, 1].tolist())
            # _pred_test_cov = pred_test_cov  # np.array(pred_test_cov[:, 1].tolist())
            #
            # """
            # ndcg_10_cov = ndcg_score(_test_data, _pred_test_cov, k=10)
            # ndcg_10_cov_females = ndcg_score(
            #     _test_data[groups[0]], _pred_test_cov[groups[0]], k=10
            # )
            # ndcg_10_cov_males = ndcg_score(
            #     _test_data[groups[1]], _pred_test_cov[groups[1]], k=10
            # )
            # """
            # ndcg_10_cov = np.mean([ndcg_score([x], [y], k=10) for x, y in zip(_test_data, _pred_test_cov)])
            # ndcg_10_cov_females = np.mean(
            #     [ndcg_score([x], [y], k=10) for x, y in
            #      zip([_test_data[idx] for idx in groups[0]], [_pred_test_cov[idx] for idx in groups[0]])]
            # )
            # ndcg_10_cov_males = np.mean(
            #     [ndcg_score([x], [y], k=10) for x, y in
            #      zip([_test_data[idx] for idx in groups[1]], [_pred_test_cov[idx] for idx in groups[1]])]
            # )
            # print(
            #     f"NDCG@10 all : {ndcg_10_cov:.5f} \t Females {ndcg_10_cov_females:.5f} \t Males {ndcg_10_cov_males:.5f} \t Chi_stat : {chi2_stat.item():.1f}"
            # )

        start_time = 0
        end_time = 0

        start_time = time.time()

        model.fit(
            X,
            nq,
            nl,
            lr=2e-2,
            cov=covariates,
            max_epoch=300,
            batch_size=(200, 200),
            scheduler_options={
                "mode": "min",
                "factor": 0.2,
                "patience": 20,
                "cooldown": 20,
            },
            #callback=callback,
            callback=None
        )

        end_time = time.time()
        execution_time = end_time - start_time

        with open(directory + "time.csv", "a") as file:
            file.write(f"{run},{execution_time}\n")

        res = {"nq": nq, "nl": nl, "model": model.to_numpy()}

        """
        res.update(
            {
                "nll": model.get_ll(X, nq, nl, cov=covariates)
                .detach()
                .cpu()
                .item()
            }
        )
        """
        # to_save.append(res)

        file_res = f"{directory}{dataset}_block_{block}_run_{run + 1}"
        if covariates is not None:
            file_res += f"_{sensitive_attribute}.pkl"
        else:
            file_res += "_baseline.pkl"

        with open(file_res, "wb") as f:
            pickle.dump(res, f)

        print(f"File saved: {file_res}")

    except Exception as e:
        print(e)

# pickle.dump(to_save, open(directory + "block_" + str(block) + f"run_{run}" + ("_age" if "age" in data_file else "") + ".pkl", "wb"))

from tauCC.src.utils_pareto import *

root = os.path.dirname(os.path.abspath(__file__))

clusters = [3]
groups = [2]
algorithm = "taucc_fair"  # "taucc_vanilla", "taucc_fair"
rc = True

for clus in clusters:
    for group in groups:
        best_parameters_fair_synth(root, clus, group, algorithm, rc)

"""
init = "random"
algorithm = "taucc_fair"  # "taucc_vanilla", "taucc_fair"

datasets = ["amazon", "lfw", "movielens-1m", "yelp", "movielens-1m"]  # "movielens-1m"
sensitive = ["gender", "gender", "gender", "gender", "age"]  # "age"
#groups = 2  #3

for idx, dataset in enumerate(datasets):
    if algorithm == "taucc_vanilla":
        best_run_vanilla(root, dataset, sensitive[idx])
    else:
        best_parameters_fair(root, dataset, sensitive[idx], algorithm=algorithm, init=init)
        #best_run_fair(root, datasets, sensitive, groups, algorithm=algorithm, init=init)
"""
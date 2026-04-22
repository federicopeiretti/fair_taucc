import os
import pandas as pd
import numpy as np
from tauCC.src.utils import aggregated_results_fair, aggregated_results_vanilla, get_aggregated, get_aggregated_synthetic

root = os.path.dirname(os.path.abspath(__file__))

init = "random"
#algorithm = "taucc_fair"  # "taucc_vanilla", "taucc_fair_max"

datasets = ["amazon", "lfw", "movielens-1m", "yelp", "movielens-1m"]  # "movielens-1m"
sensitive = ["gender", "gender", "gender", "gender", "age"]  # "age"
groups = [2, 2, 2, 2, 3]  # 3

for idx, dataset in enumerate(datasets):
    #aggregated_results_fair(dataset=dataset, sensitive=sensitive[idx], num_groups=groups[idx],
    #                        algorithm=algorithm, init=init, root=root)
    aggregated_results_vanilla(dataset=dataset, sensitive=sensitive[idx], num_run=10, init=init, root=root)
    ## SEE get_aggregated and get_aggregated_synthetic
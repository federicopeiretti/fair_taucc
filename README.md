# Fair-τCC - A fair associative co-clustering algorithm

Source code of the Fair-τCC algorithm [2], a fair version of Fast-τCC [1].

[1] Battaglia, E., Peiretti, F. & Pensa, R.G. Fast parameterless prototype-based co-clustering. Mach Learn 113, 2153–2181 (2024).

[2] Peiretti, F., Pensa, R.G. (2026). Fair Associative Co-clustering. In: Ribeiro, R.P., et al. Machine Learning and Knowledge Discovery in Databases. Research Track. ECML PKDD 2025. Lecture Notes in Computer Science(), vol 16013. Springer

## Repository Structure

The repository is organized as follows:

* `algorithms/C-Fairness-RecSys` contains the source code of the Parity LBM framework (Co-clustering for fair recommendations [Frisch et al. 2021])
* `algorithms/FATR` contains the source code of the FATR framwork (Fairness-Aware Tensor-Based Recommendation [Ziwei Zhu et al. 2018])
* `datasets` contains the datasets used for experiments. For each dataset, there are the data matrix, sensitive attribute, and ground-truth labels
* `plots` contains the plots produced during the experiments
* `results` contains the results of experiments with Fair-τCC and standard Fast-τCC
* `tauCC` contains the source code of the Fast-τCC and Fair-τCC algorithms

## Requirements

The source code contains a `requirements.txt` file that can be used to install the dependencies. The dependencies can be installed with the following command:

**Windows (Python 3):**
```
pip install -r requirements.txt
```

**Linux:**
```
pip3 install -r requirements.txt
```

## Datasets and data matrix generation

The algorithm has been tested on the following datasets:

Real-world datasets

- **MovieLens 1M** ([https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/))
- **Amazon** ([https://figshare.com/articles/dataset/Gender_Bias_In_Online_Reviews/12834617/4](https://figshare.com/articles/dataset/Gender_Bias_In_Online_Reviews/12834617/4))
- **Yelp** ([https://figshare.com/articles/dataset/Gender_Bias_In_Online_Reviews/12834617/4](https://figshare.com/articles/dataset/Gender_Bias_In_Online_Reviews/12834617/4))
- **Labeled Faces in the Wild** (downloaded using `sklearn.datasets`)

Synthetic datasets (block diagonal structure matrices for biclustering, size 1000 x 1000):

- **clus3_groups2** and **clus3_groups3**: 3 row clusters, 3 column clusters, 2 and 3 protected groups associated with row entities
- **clus5_groups2** and **clus5_groups3**: 5 row clusters, 5 column clusters, 2 and 3 protected groups associated with row entities
- **clus10_groups2** and **clus10_groups3**: : 10 row clusters, 10 column clusters, 2 and 3 protected groups associated with row entities
- **clus3_rc**: 3 row clusters, 3 column clusters, 2 protected groups associated with row entities, 2 protected groups associated with column entities

Within the `datasets` folder are the data matrices for each dataset with corresponding protected groups and true labels. Due to limited space on the free version of GitHub, we have removed the MovieLens and LFW matrices, but it is possible to generate them using the notebook in their respective folders.

### Note
For MovieLens, before generating the data matrix, it is essential to download the dataset from the official site and place the files in the `datasets/movielens/movielens-1m` folder.

## Configuration

The project includes three configuration files, each containing parameters that need to be set in order to run Fast-τCC and Fair-τCC depending on the type of dataset:
* `global_var.py` for running on real-world datasets
* `config_synthetic.py` for synthetic datasets (excluding the `clus3_rc` matrix)
* `config_synthetic_rows_cols.py` for the synthetic matrix `clus3_rc`

In particular, you should set the following parameters:

* **RUNS**: Number of runs to execute
* **ALGO_VERSION**: Version of Fair-τCC (`fair_taucc` corresponds to Fair-τCC v1, `fair_taucc_max` corresponds to Fair-τCC v2)
* **DATASET**: Name of the dataset
* **SENSITIVE**: Sensitive attribute associated with row objects
* **TRUE_LABEL**: Dataset feature considered as ground truth
* **TRUE_LABEL_DIM**: `rows` or `cols` - indicates whether the dataset feature relates to the rows or columns of the matrix
* **fair_majority_range**: Alpha values in [0.0, 1.0] for the majority group
* **fair_minority_range**: Alpha values in [0.0, 1.0] for the minority group
* **fair_minority_range2**: Alpha values in [0.0, 1.0] for the second minority group (the least represented in the dataset). Used only with 3 protected groups.

## Running the Algorithm

To run the algorithms, use the following commands:

### Fast-τCC algorithm
```bash
fair_taucc/E1_vanilla.py                  # real-world datasets
fair_taucc/E2_vanilla_synthetic.py        # synthetic datasets (clus3, clus5, clus10)
fair_taucc/E3_vanilla_synth_rc.py         # synthetic dataset (clus3_rc)
```

### Fair-τCC algorithm with 2 protected groups
```bash
fair_taucc/E1_fair_2groups.py                 # real-world datasets
fair_taucc/E2_fair_2groups_synthetic.py       # synthetic datasets (clus3, clus5, clus10)
fair_taucc/E3_fair_2groups_synthetic_rc.py    # synthetic datasets (clus3_rc)
```

### Fair-τCC algorithm with 3 protected groups
```bash
fair_taucc/E1_fair_3groups.py                 # real-world datasets
fair_taucc/E2_fair_3groups_synthetic.py       # synthetic datasets (clus3, clus5, clus10)
fair_taucc/E3_fair_3groups_synthetic_rc.py    # synthetic datasets (clus3_rc)
```

### Parity LBM (Frisch et al.)

The project root is `algorithms/C-Fairness-RecSys`.

To run the Parity LBM algorithm, follow these steps:

1. Create a virtual environment with Python 3.8 and install the dependencies in the `requirements.txt` files:
   - `requirements.txt` in the project root
   - `reproducibility_study/Frisch_et_al/requirements.txt`
2. Generate the required `.pkl` files using the notebooks in the `algorithms/C-Fairness-RecSys` path:
   - `preprocessed_datasets.ipynb` for 2 protected groups
   - `preprocessed_datasets_3groups.ipynb` for 3 protected groups
3. Edit the following parameters in the `start_experiments.py` file:
   - **dataset**: dataset name
   - **sensitive_attribute**: sensitive feature associated with row objects
   - **covariates**: set `covariates = None` (line 117) to compute baseline, otherwise leave the line commented
   - **nq, nl**: number of row/column clusters to be found

#### Command for running the algorithm on a dataset with 2 protected groups:
```bash
python3 start_experiments.py > logs/movielens_gender.log
```

#### Command for running the algorithm on a dataset with 3 protected groups:
```bash
python3 start_experiments_3groups.py > logs/movielens_age.log
```


## Original Algorithms Code

The original code of TauCC algorithm is property of Elena Battaglia, Federico Peiretti and Ruggero G. Pensa.  
Link: [https://github.com/rupensa/tauCC/](https://github.com/rupensa/tauCC/)

The original code of Parity LBM framwork is property of Gabriel Frisch, Jean-Benoist Leger and Yves Grandvalet.  
Link: [https://github.com/jackmedda/C-Fairness-RecSys/tree/main/reproducibility_study/Frisch_et_al](https://github.com/jackmedda/C-Fairness-RecSys/tree/main/reproducibility_study/Frisch_et_al)

The original code of FATR framework is property of Ziwei Zhu, Xia Hu, and James Caverlee.  
Link: [https://github.com/Zziwei/Fairness-Aware_Tensor-Based_Recommendation](https://github.com/Zziwei/Fairness-Aware_Tensor-Based_Recommendation)

## Citation

```bibtex
@inproceedings{DBLP:conf/pkdd/PeirettiP25,
  author       = {Federico Peiretti and
                  Ruggero G. Pensa},
  editor       = {Rita P. Ribeiro and
                  Bernhard Pfahringer and
                  Nathalie Japkowicz and
                  Pedro Larra{\~{n}}aga and
                  Al{\'{\i}}pio M. Jorge and
                  Carlos Soares and
                  Pedro H. Abreu and
                  Jo{\~{a}}o Gama},
  title        = {Fair Associative Co-clustering},
  booktitle    = {Machine Learning and Knowledge Discovery in Databases. Research Track
                  - European Conference, {ECML} {PKDD} 2025, Porto, Portugal, September
                  15-19, 2025, Proceedings, Part {I}},
  series       = {Lecture Notes in Computer Science},
  pages        = {282--300},
  publisher    = {Springer},
  year         = {2025},
  url          = {https://doi.org/10.1007/978-3-032-05962-8\_17},
  doi          = {10.1007/978-3-032-05962-8\_17},
  timestamp    = {Sun, 09 Nov 2025 16:31:12 +0100},
  biburl       = {https://dblp.org/rec/conf/pkdd/PeirettiP25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

# Fair τCC

Source code of the Fair TauCC algorithm, a fair version of Fast τCC proposed by Battaglia et al. in [1].

[1] Battaglia, E., Peiretti, F., Pensa, R.G.: Fast parameterless prototype-based co-clustering. Mach. Learn. 113(4), 2153–2181 (2024)

## Repository Structure

The repository is organized as follows:

* `algorithms/C-Fairness-RecSys` contains the source code of the Parity LBM algorithm
* `datasets` contains the datasets used for experiments. For each dataset, there are the data matrix, sensitive attribute, and ground-truth labels
* `results` contains the results of experiments with Fair TauCC algorithm and standard TauCC version
* `tauCC` contains the source code of the TauCC and Fair TauCC algorithms

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

- **MovieLens 1M** ([https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/))
- **Amazon** ([https://figshare.com/articles/dataset/Gender_Bias_In_Online_Reviews/12834617/4](https://figshare.com/articles/dataset/Gender_Bias_In_Online_Reviews/12834617/4))
- **Yelp** ([https://figshare.com/articles/dataset/Gender_Bias_In_Online_Reviews/12834617/4](https://figshare.com/articles/dataset/Gender_Bias_In_Online_Reviews/12834617/4))
- **Labeled Faces in the Wild** (downloaded using `sklearn.datasets`)

Within the `datasets` folder are the data matrices for each dataset with corresponding protected groups and true labels. Due to limited space on the free version of GitHub, we have removed the MovieLens and LFW matrices, but it is possible to generate them using the notebook in their respective folders.

### Note
For MovieLens, before generating the data matrix, it is essential to download the dataset from the official site and place the files in the `datasets/movielens/movielens-1m` folder.

## Configuration

The `global_var.py` file is a configuration file containing parameters that must be set before running the TauCC and Fair TauCC algorithms. In particular, you should set the following parameters:

* **RUNS**: Number of runs to execute
* **DATASET**: Name of the dataset
* **SENSITIVE**: Sensitive attribute associated with row objects
* **TRUE_LABEL**: Dataset feature considered as ground truth
* **TRUE_LABEL_DIM**: `rows` or `cols` - indicates whether the dataset feature relates to the rows or columns of the matrix
* **fair_majority_range**: Alpha values in [0.0, 1.0] for the majority group
* **fair_minority_range**: Alpha values in [0.0, 1.0] for the minority group
* **fair_minority_range2**: Alpha values in [0.0, 1.0] for the second minority group (the least represented in the dataset). Used only with 3 protected groups.

## Running the Algorithm

To run the algorithms, use the following commands:

### Vanilla TauCC algorithm
```bash
fair_taucc/E1_vanilla.py > fair_taucc/logs/TauCC_MovieLens_gender.log
```

### Fair TauCC algorithm with 2 protected groups
```bash
fair_taucc/E1_fair_2groups.py > fair_taucc/logs/E1_fair_2groups_MovieLens1M_gender.log
```

### Fair TauCC algorithm with 3 protected groups
```bash
fair_taucc/E1_fair_3groups.py > fair_taucc/logs/E1_fair_3groups_MovieLens1M_age.log
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

The original code of Parity LBM algorithm is property of Gabriel Frisch, Jean-Benoist Leger and Yves Grandvalet.  
Link: [https://github.com/jackmedda/C-Fairness-RecSys/tree/main/reproducibility_study/Frisch_et_al](https://github.com/jackmedda/C-Fairness-RecSys/tree/main/reproducibility_study/Frisch_et_al)


## Citation

<pre>
   Fair Associative Co-Clustering
   F. Peiretti and R. G. Pensa
   European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 
   ECML PKDD 2025
   Porto, Portugal
   September 15th - 19th 2025
</pre>

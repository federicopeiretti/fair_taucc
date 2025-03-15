# Fair τCC

Source code of the Fair TauCC algorithm, a fair version of Fast τCC proposed by Battaglia et al. in [1].

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

## References

```bibtex
@article{Battaglia2024,
  title={Fast parameterless prototype-based co-clustering},
  author={Elena Battaglia and Federico Peiretti and Ruggero G. Pensa},
  journal={Mach. Learn.},
  year={2024},
  volume = {113},
  number = {4},
  pages={2153--2181}
}
```

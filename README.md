# Collaborative Filtering

A package for collaborative filtering recommendation.

**Currently supported models**

- MF: matrix factorization
- MLP: multilayer perceptron

**Currently supported loss types**

- BCE: binary cross-entropy
- CE: cross-entropy (pseudo multiclass classification)
- BPR: Bayesian personalized ranking
- GBPR: group Bayesian personalized ranking

## Requirements

- Linux-based OS
- Python 3.6+


## Get started

### Install the package

Install from <https://pypi.org/>:

```bash
pip install collaborative-filtering
```

Or install it manually:

```bash
git clone https://github.com/yusanshi/collaborative-filtering.git
cd collaborative-filtering
pip install .
```

### Prepare the dataset

You should specify the dataset by passing the `--dataset_path` parameter. The parameter value must be the directory path with `train.tsv`, `valid.tsv` and `test.tsv` in it. Each file must be a two-columns TSV file with `user` and `item` as headers, user indexs and item indexs (both are 0-based) as body. An example:
```tsv
user	item
0	241
0	149
0	76
...
```

Based on the datasets used in [GBPR](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.415.9378&rep=rep1&type=pdf), we made some small changes on data format so you can directly use them in this project. Download and uncompress them with:
```bash
mkdir data && cd data
wget https://github.com/yusanshi/collaborative-filtering/files/7052355/dataset.tar.gz
tar -xzvf dataset.tar.gz
```

Besides the `--dataset_path` parameter, you should also provide the `--user_num` and `--item_num` parameters. For users, make sure this formula holds: `user_num >= max(max(training user indexs), max(validation user indexs), max(test user indexs)) + 1`. The same for items. Based on the formulas, you can write a simple script to get the `user_num` and `item_num` for a dataset.

> For Quick Start users: If you use the datasets provided by us, for ML100K dataset, they are `--user_num 943 --item_num 1682`.

### Run

```bash
python -m collaborative_filtering.train \
  --user_num USER_NUM \
  --item_num ITEM_NUM \
  --negative_sampling_ratio NEGATIVE_SAMPLING_RATIO \
  --model_name {MF,MLP} \
  --loss_type {BCE,CE,BPR,GBPR} \
  --dataset_path DATASET_PATH \
  ...
```
Here we only list the most important parameters. For more details refer to `python -m collaborative_filtering.train -h` and `collaborative_filtering/parameters.py`.

## TODO

- [ ] More models
- [ ] More loss types
- [ ] Test
- [ ] Documentation


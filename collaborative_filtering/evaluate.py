import numpy as np
import pandas as pd
import torch

from os import path
from tqdm import tqdm
from multiprocessing import Pool
from .metrics import calculate_metric
from .parameters import parse_args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()

# A simple cache mechanism for df reading and processing, since it will be run for many times
_evaluation_cache = {}


def calculate_single_user_metric(pair):
    return [
        calculate_metric(*pair, metric) for metric in args.evaluation_metrics
    ]


@torch.no_grad()
def evaluate(model, evaluation_mode, logger):
    assert evaluation_mode in ['valid', 'test']

    metrics = {}
    evaluation_file_path = path.join(args.dataset_path,
                                     f'{evaluation_mode}.tsv')
    if evaluation_mode == 'valid' and evaluation_file_path in _evaluation_cache:
        user2ranking_list = _evaluation_cache[evaluation_file_path]
    else:
        train_df = pd.read_table(path.join(args.dataset_path,
                                           'train.tsv')).drop_duplicates()
        assert train_df.columns.tolist() == ['user', 'item']
        train_user2positive = train_df.groupby('user',
                                               sort=False)['item'].apply(set)
        evaluation_df = pd.read_table(evaluation_file_path).drop_duplicates()
        assert evaluation_df.columns.tolist() == ['user', 'item']
        evaluation_user2positive = evaluation_df.groupby(
            'user', sort=False)['item'].apply(set)
        all_items = set(range(args.item_num))

        user2ranking_list = {}
        for user in range(0, args.user_num):
            try:
                true_items = evaluation_user2positive[
                    user] - train_user2positive[user]
                false_items = all_items - train_user2positive[
                    user] - evaluation_user2positive[user]
                user2ranking_list[user] = {
                    True: true_items,
                    False: false_items
                }
            except KeyError:
                pass
        if len(user2ranking_list) < args.user_num:
            logger.warning(
                f'Drop {args.user_num - len(user2ranking_list)} users in evaluation'
            )

        if evaluation_mode == 'valid':
            _evaluation_cache[evaluation_file_path] = user2ranking_list

    tasks = []
    for user, ranking_list in tqdm(user2ranking_list.items(),
                                   desc='Evaluating users'):
        y_true = np.array([1] * len(ranking_list[True]) +
                          [0] * len(ranking_list[False]))
        item_indexs = torch.tensor(
            list(ranking_list[True]) + list(ranking_list[False])).to(device)
        user_indexs = torch.tensor(user).expand_as(item_indexs).to(device)
        y_pred = model(user_indexs, item_indexs)
        y_pred = y_pred.cpu().numpy()
        tasks.append((y_true, y_pred))

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)
    metrics = {
        k: v
        for k, v in zip(args.evaluation_metrics,
                        np.array(results).mean(axis=0))
    }

    overall = np.mean(list(metrics.values()))
    return metrics, overall

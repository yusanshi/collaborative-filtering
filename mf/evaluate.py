import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from mf.metrics import calculate_metric
from mf.parameters import parse_args
from multiprocessing import Pool

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()

# A simple cache mechanism for df reading and processing, since it will be run for many times
_evaluation_cache = {}


def calculate_single_user_metric(pair):
    return [
        calculate_metric(*pair, metric) for metric in args.evaluation_metrics
    ]


@torch.no_grad()
def evaluate(model, evaluation_data_path, logger):
    metrics = {}
    if 'valid' in evaluation_data_path and evaluation_data_path in _evaluation_cache:
        user2ranking_list = _evaluation_cache[evaluation_data_path]
    else:
        train_df = pd.read_table(args.train_data_path,
                                 sep=' ',
                                 names=['user', 'item']).drop_duplicates() - 1

        train_user2positive = train_df.groupby('user',
                                               sort=False)['item'].apply(set)
        evaluation_df = pd.read_table(
            evaluation_data_path, sep=' ', names=['user', 'item'
                                                  ]).drop_duplicates() - 1
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

        if 'valid' in evaluation_data_path:
            _evaluation_cache[evaluation_data_path] = user2ranking_list

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

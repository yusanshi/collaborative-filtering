import numpy as np
import pandas as pd
import time
import torch
import os
import logging
import coloredlogs
import math
import datetime
import copy

from mf.utils.metrics import *
from mf.parameters import parse_args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = parse_args()

# A simple cache mechanism for df reading and sorting, since it will be run for many times
_df_cache_for_validation = {}


@torch.no_grad()
def evaluate(model, data_path):
    metrics = {}
    if 'valid' in data_path and data_path in _df_cache_for_validation:
        df = _df_cache_for_validation[data_path]
    else:
        df = pd.read_table(data_path, sep=' ', names=['user', 'item']) - 1
        df.sort_values(df.columns[0], inplace=True)
        if 'valid' in data_path:
            _df_cache_for_validation[data_path] = df

    columns = df.columns.tolist()
    test_data = np.transpose(df.values)
    test_data = torch.from_numpy(test_data).to(device)
    first_indexs, second_indexs, y_trues = test_data
    y_preds = []
    y_trues = y_trues.cpu().numpy()
    for i in range(math.ceil(len(df) / (8 * args.batch_size))):
        first_index = first_indexs[i * (8 * args.batch_size):(i + 1) *
                                   (8 * args.batch_size)]
        second_index = second_indexs[i * (8 * args.batch_size):(i + 1) *
                                     (8 * args.batch_size)]
        first = {'name': columns[0], 'index': first_index}
        second = {'name': columns[1], 'index': second_index}
        y_pred = model(first, second, task['name'])
        y_pred = y_pred.cpu().numpy()
        y_preds.append(y_pred)

    y_preds = np.concatenate(y_preds, axis=0)

    single_sample_length = df.groupby(columns[0]).size().values
    assert len(
        set(single_sample_length)
    ) == 1, f'The number of {columns[1]}s for different {columns[0]}s should be equal'
    y_trues = y_trues.reshape(-1, single_sample_length[0])
    y_preds = y_preds.reshape(-1, single_sample_length[0])
    metrics = {
        'AUC': fast_roc_auc_score(y_trues,
                                  y_preds,
                                  num_processes=args.num_workers),
        'MRR': mrr(y_trues, y_preds),
        'NDCG@10': ndcg_score(y_trues, y_preds, k=10, ignore_ties=True),
        'NDCG@50': ndcg_score(y_trues, y_preds, k=50, ignore_ties=True),
        'Recall@10': recall(y_trues, y_preds, k=10),
        'Recall@50': recall(y_trues, y_preds, k=50),
    }

    overall = np.mean(list(metrics.values()))
    return metrics, overall


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory) if 'keep' not in x
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def create_logger():
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG',
                        logger=logger,
                        fmt='%(asctime)s %(levelname)s %(message)s')
    log_dir = os.path.join(args.log_path,
                           f'TODO{get_dataset_name(args.train_data_path)}')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_dir,
        f"{str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')}{'-remark-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}.txt"
    )
    logger.info(f'Check {log_file_path} for the log of this run')
    file_hander = logging.FileHandler(log_file_path)
    logger.addHandler(file_hander)
    return logger


def copy_arguments(f):
    def selectively_copy(x):
        if isinstance(x, list) or isinstance(x, dict):
            return copy.deepcopy(x)
        else:
            return x

    def wrapper(*args, **kwargs):
        args = tuple(selectively_copy(x) for x in args)
        kwargs = {k: selectively_copy(v) for k, v in kwargs.items()}
        return f(*args, **kwargs)

    return wrapper


@copy_arguments
def deep_apply(d, f=lambda x: f'{x:.4f}'):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = deep_apply(v, f)
        else:
            d[k] = f(v)
    return d


def dict2table(d, k_fn=str, v_fn=lambda x: f'{x:.4f}'):
    '''
    Convert a nested dict to markdown table
    '''
    def parse_header(d, depth=0):
        if isinstance(list(d.values())[0], dict):
            header = parse_header(list(d.values())[0], depth=depth + 1)
            for v in d.values():
                assert header == parse_header(v, depth=depth + 1)
            return header
        else:
            return f"| {' | '.join([''] * depth + list(map(k_fn, d.keys())))} |"

    def parse_content(d, accumulated_keys=[]):
        if isinstance(list(d.values())[0], dict):
            contents = []
            for k, v in d.items():
                contents.extend(parse_content(v, accumulated_keys + [k_fn(k)]))
            return contents
        else:
            return [
                f"| {' | '.join(accumulated_keys + list(map(v_fn, d.values())))} |"
            ]

    lines = [parse_header(d), *parse_content(d)]
    return '\n'.join(lines)


def get_dataset_name(train_data_path):
    filename_parts = train_data_path.split('/')[-1].split('-')
    assert len(filename_parts) == 3
    return '-'.join(filename_parts[:2])

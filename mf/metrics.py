import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score


def recall_score(y_trues, y_scores, k):
    assert y_trues.shape == y_scores.shape
    assert len(y_trues.shape) == 2
    orders = np.argsort(y_scores, axis=-1)[:, ::-1][:, :k]
    return np.mean(
        np.sum(np.take_along_axis(y_trues, orders, axis=-1), axis=-1) /
        np.sum(y_trues, axis=-1))


def mrr_score(y_trues, y_scores):
    assert y_trues.shape == y_scores.shape
    assert len(y_trues.shape) == 2
    orders = np.argsort(y_scores, axis=-1)[:, ::-1]
    y_trues = np.take_along_axis(y_trues, orders, axis=-1)
    rr_scores = y_trues / (np.arange(y_trues.shape[1]) + 1)
    return np.mean(np.sum(rr_scores, axis=-1) / np.sum(y_trues, axis=-1))


def calculate_metric(y_true, y_pred, metric):
    '''
    Args:
        y_true, y_pred: 1-d numpy array
        metric: metric name
    '''
    metric = metric.lower()
    if metric == 'auc':
        return roc_auc_score(y_true, y_pred)
    if metric == 'mrr':
        return mrr_score(y_true[np.newaxis, :], y_pred[np.newaxis, :])
    if metric.startswith('ndcg@'):
        k = int(metric.split('@')[-1])
        return ndcg_score(y_true[np.newaxis, :],
                          y_pred[np.newaxis, :],
                          k=k,
                          ignore_ties=True)
    if metric.startswith('recall@'):
        k = int(metric.split('@')[-1])
        return recall_score(y_true[np.newaxis, :], y_pred[np.newaxis, :], k=k)

    raise NotImplementedError

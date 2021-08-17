from torch.utils.data import Dataset
from os import path

import pandas as pd
import numpy as np
import random


class TrainingDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        df_positive = pd.read_table(
            path.join(args.dataset_path,
                      'train.tsv')).drop_duplicates().sort_values('user')
        assert df_positive.columns.tolist() == ['user', 'item']
        positive_set = set(map(tuple, df_positive.values))
        user_indexs = df_positive['user'].values
        positive_num_map = df_positive.groupby('user').size().to_dict()
        positive_item_indexs = df_positive['item'].values
        negative_item_indexs = np.concatenate([
            random.sample(range(args.item_num),
                          v * args.negative_sampling_ratio)
            for v in positive_num_map.values()
        ],
                                              axis=0)

        negative_set = set(
            zip(user_indexs.repeat(args.negative_sampling_ratio),
                negative_item_indexs))
        common = positive_set & negative_set
        negative_set = negative_set - positive_set
        for e in common:
            user_index = e[0]
            while True:
                pair = (user_index, random.choice(range(args.item_num)))
                if pair not in negative_set and pair not in positive_set:
                    negative_set.add(pair)
                    break

        negative_item_indexs = pd.DataFrame(
            np.array([*negative_set]),
            columns=['user',
                     'item']).sort_values('user')['item'].values.reshape(
                         user_indexs.shape[0], args.negative_sampling_ratio)
        self.user_indexs = user_indexs
        self.positive_item_indexs = positive_item_indexs
        self.negative_item_indexs = negative_item_indexs

        if args.loss_type == 'GBPR':
            self.group_user_indexs = np.stack([
                random.sample(range(args.user_num), args.group_size)
                for _ in range(len(user_indexs))
            ])

    def __len__(self):
        return len(self.user_indexs)

    def __getitem__(self, index):
        return (
            self.user_indexs[index],
            self.positive_item_indexs[index],
            self.negative_item_indexs[index],
            self.group_user_indexs[index]
            if self.args.loss_type == 'GBPR' else 0,
        )


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = {
        'dataset_path': 'data/ML100K-copy1',
        'user_num': 943,
        'item_num': 1682,
        'negative_sampling_ratio': 4,
        'loss_type': 'BPR',
        'group_size': 3
    }

    dataset = TrainingDataset(dotdict(args))
    dataloader = DataLoader(dataset,
                            batch_size=16,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)
    for minibatch in dataloader:
        print(minibatch)
        break

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
        user_indexes = df_positive['user'].values
        positive_num_map = df_positive.groupby('user').size().to_dict()
        positive_item_indexes = df_positive['item'].values
        negative_item_indexes = np.concatenate([
            random.sample(range(args.item_num),
                          v * args.negative_sampling_ratio)
            for v in positive_num_map.values()
        ],
                                               axis=0)

        negative_set = set(
            zip(user_indexes.repeat(args.negative_sampling_ratio),
                negative_item_indexes))
        common = positive_set & negative_set
        negative_set = negative_set - positive_set
        for e in common:
            user_index = e[0]
            while True:
                pair = (user_index, random.choice(range(args.item_num)))
                if pair not in negative_set and pair not in positive_set:
                    negative_set.add(pair)
                    break

        negative_item_indexes = pd.DataFrame(
            np.array([*negative_set]),
            columns=['user',
                     'item']).sort_values('user')['item'].values.reshape(
                         user_indexes.shape[0], args.negative_sampling_ratio)
        self.user_indexes = user_indexes
        self.positive_item_indexes = positive_item_indexes
        self.negative_item_indexes = negative_item_indexes

        if args.loss_type == 'GBPR':
            self.group_user_indexes = np.stack([
                random.sample(range(args.user_num), args.group_size)
                for _ in range(len(user_indexes))
            ])

    def __len__(self):
        return len(self.user_indexes)

    def __getitem__(self, index):
        return (
            self.user_indexes[index],
            self.positive_item_indexes[index],
            self.negative_item_indexes[index],
            self.group_user_indexes[index]
            if self.args.loss_type == 'GBPR' else 0,
        )

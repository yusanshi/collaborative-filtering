import unittest

from collaborative_filtering.dataset import TrainingDataset
from torch.utils.data import DataLoader


class TestDataset(unittest.TestCase):
    def test_dataset_reading(self):
        class dotdict(dict):
            """dot.notation access to dictionary attributes"""
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        args = {
            'dataset_path': 'data/ML100K-copy1',
            'user_num': 943,
            'item_num': 1682,
            'negative_sampling_ratio': 1,
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
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()

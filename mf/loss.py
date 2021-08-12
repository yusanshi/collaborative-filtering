import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positive_score, negative_score):
        '''
        Args:
            positive_score: batch_size
            negative_score: batch_size
        '''
        return -F.logsigmoid(positive_score - negative_score).mean()


class GBPRLoss(nn.Module):
    def __init__(self, group_coefficient_rho):
        super().__init__()
        self.group_coefficient_rho = group_coefficient_rho

    def forward(self, positive_score, negative_score, group_positive_score):
        '''
        Args:
            positive_score: batch_size
            negative_score: batch_size
            group_positive_score: batch_size, group_size
        '''
        group_positive_score = group_positive_score.mean(dim=-1)
        positive_score = self.group_coefficient_rho * group_positive_score + (
            1 - self.group_coefficient_rho) * positive_score
        return -F.logsigmoid(positive_score - negative_score).mean()

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.user_embedding = nn.Embedding(args.user_num, args.embedding_dim)
        self.item_embedding = nn.Embedding(args.item_num, args.embedding_dim)
        # Not use dot product but use a linear layer instead. See GMF in [NCF](https://arxiv.org/abs/1708.05031) by Xiangnan He et al.
        self.linear = nn.Linear(args.embedding_dim, 1, bias=False)

    def forward(self, user_index, item_index):
        # batch_size
        return self.linear(
            torch.mul(
                self.user_embedding(user_index),
                self.item_embedding(item_index),
            )).squeeze(dim=1)

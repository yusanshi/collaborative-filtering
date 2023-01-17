import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.user_embedding = nn.Embedding(args.user_num, args.embedding_dim)
        self.item_embedding = nn.Embedding(args.item_num, args.embedding_dim)
        self.MLP = nn.Sequential(
            nn.Linear(2 * args.embedding_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, args.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(args.embedding_dim // 2, 1, bias=False),
        )

    def forward(self, user_index, item_index):
        if self.args.random_user:
            user_index = torch.randint(self.args.user_num,
                                       size=user_index.shape,
                                       device=user_index.device)
        # batch_size
        return self.MLP(
            torch.cat((self.user_embedding(user_index),
                       self.item_embedding(item_index)),
                      dim=-1)).squeeze(dim=1)

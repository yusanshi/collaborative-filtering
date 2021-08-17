import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
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
        # batch_size
        return self.MLP(
            torch.cat((self.user_embedding(user_index),
                       self.item_embedding(item_index)),
                      dim=-1)).squeeze(dim=1)

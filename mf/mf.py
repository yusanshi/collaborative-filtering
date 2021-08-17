import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.user_embedding = nn.Embedding(args.user_num, args.embedding_dim)
        self.item_embedding = nn.Embedding(args.item_num, args.embedding_dim)
        self.linear = nn.Linear(args.embedding_dim, 1,
                                bias=False)  # TODO bias?

    def forward(self, user_index, item_index):
        # batch_size, embedding_dim
        vector = torch.mul(
            self.user_embedding(user_index),
            self.item_embedding(item_index),
        )
        # batch_size
        score = self.linear(vector).squeeze(dim=1)
        return score

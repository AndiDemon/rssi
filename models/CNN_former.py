import torch
import torch.nn as nn


class CNNformer(nn.Module):
    def __init__(self, input_dim=3, conv_hidden=32, lin_hidden=32, num_heads=3, drop=0.5):
        super(CNNformer, self).__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv1d(3, conv_hidden, 1, 1),
            nn.ReLU(),
            nn.Conv1d(conv_hidden, 3, 1, 1),
            nn.BatchNorm1d(input_dim),
        )

        self.mha = nn.MultiheadAttention(embed_dim=9, num_heads=3)
        self.norm = nn.BatchNorm1d(9)

        self.mlp = nn.Sequential(
            nn.Linear(9, lin_hidden),
            nn.ReLU(),
            nn.Linear(lin_hidden, 9),
        )
        self.mlp_out = nn.Linear(9, 2)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.cnn_block(x)

        x = torch.flatten(x, 1)

        for i in range(3):
            y, y_weight = self.mha(x, x, x)
            x = x + self.norm(y)


        x = self.mlp(x)
        x = self.mlp_out(x)
        x = self.drop(x)

        return x

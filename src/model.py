import torch
import torch.nn as nn

import config


class MLP(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, hidden=config.HIDDEN_DIMS,
                 num_classes=config.NUM_CLASSES, dropout=config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], num_classes),
        )

    def forward(self, x):
        # log-softmax + NLLLoss, as in the original code
        return torch.log_softmax(self.net(x), dim=1)


def build_model():
    return MLP()

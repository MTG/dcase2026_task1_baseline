import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.01)

    def forward(self, logits, labels):
        return self.cross_entropy(logits, labels)
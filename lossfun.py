from __future__ import print_function
import torch


class MixLoss(object):
    def __call__(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1)
        loss = torch.mean((probs - targets) ** 2)
        return loss

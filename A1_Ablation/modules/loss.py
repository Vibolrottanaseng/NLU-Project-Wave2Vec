import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask):

        logits = logits[:, :targets.size(1), :]
        targets = targets[:, :logits.size(1)]
        mask = mask[:, :logits.size(1)]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0,
            reduction='none'
        )

        loss = loss.view(targets.size()) * mask

        return loss.sum() / (mask.sum() + 1e-8)


def compute_loss(logits, targets, masks):
    criterion = LanguageModelCriterion()
    return criterion(logits, targets[:, 1:], masks[:, 1:])
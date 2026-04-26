import torch
import torch.nn as nn

class ProblemEncoder(nn.Module):
    def __init__(self, input_dim=1357, embed_dim=2048):
        super(ProblemEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        print("ProblemEncoder input dim:", self.fc.in_features)

    def forward(self, x):
        return self.fc(x)
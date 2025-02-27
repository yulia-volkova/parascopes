import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.linear = nn.Linear(conf.d_res, conf.d_sonar)

    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # More balanced hidden layer dimensions based on input/output sizes
        self.d_res = conf.d_res
        self.d_mlp = conf.d_mlp
        self.d_sonar = conf.d_sonar
        self.sequential = nn.Sequential(
            nn.Linear(self.d_res, self.d_mlp),
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(self.d_mlp, self.d_mlp),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_mlp, self.d_sonar)
        )

    def forward(self, x):
        return self.sequential(x)
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.linear = nn.Linear(conf.d_res, conf.d_sonar)

    def forward(self, x):
        return self.linear(x)

class ScaledLinear(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # Create a scale parameter for each input dimension
        self.scale = nn.Parameter(torch.ones(conf.d_res))
        self.linear = nn.Linear(conf.d_res, conf.d_sonar)

    def forward(self, x):
        # Scale each input dimension independently
        # Reshape scale to match batch dimension of x
        scale = self.scale
        while len(scale.shape) < len(x.shape):
            scale = scale.unsqueeze(0)
        scaled_x = x * scale
        return self.linear(scaled_x)

    def normalize(self):
        """
        Normalizes the linear layer weights and adjusts the scale parameters accordingly.
        This moves the magnitude from the weights to the scale component for each input dimension.
        """
        with torch.no_grad():
            # Get the weight matrix
            weight = self.linear.weight  # shape: [d_sonar, d_res]

            # Calculate the norm of each input dimension (column)
            weight_norms = torch.norm(weight, dim=0)  # shape: [d_res]

            # Create a mask for non-zero norms to avoid division by zero
            mask = weight_norms > 0

            # Normalize each column of the weight matrix where norm > 0
            normalized_weight = weight.clone()
            normalized_weight[:, mask] = normalized_weight[:, mask] / weight_norms[mask].unsqueeze(0)
            self.linear.weight.data = normalized_weight

            # Move the scaling factors to the input scale parameters
            self.scale.data[mask] = self.scale.data[mask] * weight_norms[mask]

            # Also adjust the bias if it exists
            if self.linear.bias is not None:
                # No need to adjust bias as it's added after the linear transformation
                pass

        return self




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
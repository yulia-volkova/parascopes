# %%
from utils_train import Trainer
import torch
import numpy as np
import einops
import matplotlib.pyplot as plt
from transformers import AutoConfig
torch.set_grad_enabled(False)


cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = Trainer.load_from_wandb("notebooks-sonar/4nbwxrar")
w = model.model.linear.weight.cpu()


# %%
w_l = einops.rearrange(w, "d_sonar (n_layers d_model) -> n_layers d_sonar d_model", d_sonar=1024, d_model=3072)
print(w_l.shape, w_l.shape[0]//2, "out of", cfg.num_hidden_layers, "layers")

# Instead of Frobenius norm, try these alternatives:
# 1. L1 norm (sum of absolute values) - emphasizes sparsity
# w_l_norm = torch.abs(w_l).sum(dim=(1, 2))

# 2. Per-neuron L2 norm (analyze individual dimensions)
# w_l_norm = torch.linalg.norm(w_l, dim=2, ord=2)  # Shape: [n_layers, d_sonar]

# 3. Max absolute value (find most influential weights)
#w_l_norm = torch.max(torch.abs(w_l), dim=2)[0].mean(dim=1)

# 4. Frobenius norm
w_l_norm = torch.linalg.norm(w_l, dim=(1, 2), ord='fro')

total_layers = cfg.num_hidden_layers
n_layers = w_l.shape[0]
layer_names = [f"{'attn' if i % 2 == 0 else 'mlp'} {total_layers - n_layers//2 + i//2}" for i in range(n_layers)]
print(w_l_norm)

# bar chart of the norms
plt.figure(figsize=(10, 5))
bars = plt.bar(layer_names, w_l_norm, color=['darkblue' if i % 2 == 0 else 'darkred' for i in range(len(layer_names))])
plt.xticks([])  # Remove x-axis labels since we'll put them in bars

height = np.mean([bar.get_height() for bar in bars])

# Add labels inside the bars
for bar, name, norm in zip(bars, layer_names, w_l_norm):
    plt.text(bar.get_x() + bar.get_width()*0.55, height/2,
             f"{name}  ({norm:.2f})",
             ha='center', va='center',
             rotation=90,
             color='white')

plt.ylabel("Frobenius Norm")

plt.show()

# %%

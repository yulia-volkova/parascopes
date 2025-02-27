# %%
import taker
from taker import Model
import json
import torch
from tqdm import tqdm
torch.set_grad_enabled(False)

# %%
DATASET_FILE = '/workspace/SPAR/gen-dataset/split_indexed_dataset.jsonl'
with open(DATASET_FILE, 'r') as file:
    dataset = [json.loads(line) for line in file]

print("Dataset loaded:", len(dataset))
print(dataset[0])


m = Model("meta-llama/Llama-3.2-3B-Instruct", dtype="hqq8", limit=64000)


# %%
import os
os.makedirs("./tensors", exist_ok=True)

import os

# Determine how many files exist in ./tensors
existing_files = len([name for name in os.listdir("./tensors") if name.startswith("res_data_") and name.endswith(".pt")])
skip_count = existing_files * 1000

batch = []
for i, data in enumerate(tqdm(dataset)):
    if i < skip_count:
        continue
    if i > 0 and i % 1000 == 0:
        batch_index = i // 1000 - 1
        torch.save(batch, f"./tensors/res_data_{batch_index:03d}.pt")
        batch = []

    text = "".join(data["split"])
    inputs, attns, mlps, output = m.get_residual_diffs(text)

    # Interweave the tensors as specified
    res = [inputs[:, None]]
    max_len = min(attns.shape[1], mlps.shape[1])  # Ensure we don't exceed the dimensions
    for i in range(max_len):
        res.append(attns[:, i, None])
        res.append(mlps[:, i, None])
    res = torch.cat(res, dim=1)

    indices = torch.tensor([i-1 for i in data["indices"]])

    good_res = res[:, :, indices]

    batch.append(good_res)

batch_index += 1
torch.save(batch, f"./tensors/res_data_{batch_index:03d}.pt")

# %%

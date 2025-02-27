# %%
import taker
from taker import Model
import json
import torch
from tqdm import tqdm
import os
import gc

torch.set_grad_enabled(False)
# %%
# Load dataset
with open('/workspace/SPAR/gen-dataset/split_indexed_dataset.jsonl', 'r') as file:
    dataset = [json.loads(line) for line in file]

# Setup model
m = Model("meta-llama/Llama-3.2-3B-Instruct", dtype="hqq8", limit=64000)

# Create tensors directory
os.makedirs("./tensors", exist_ok=True)

# Calculate expected files based on dataset size
total_files = (len(dataset) + 999) // 1000

# Get existing files
existing_files = {int(name.split('_')[2].split('.')[0]) for name in os.listdir("./tensors")
                 if name.startswith("res_data_") and name.endswith(".pt")}

# %%
# Process missing files
for batch_index in tqdm(range(total_files)):
    if batch_index in existing_files:
        continue

    start_idx = batch_index * 1000
    end_idx = min(start_idx + 1000, len(dataset))
    batch = []
    gc.collect()
    torch.cuda.empty_cache()

    for data in dataset[start_idx:end_idx]:
        text = "".join(data["split"])
        inputs, attns, mlps, output = m.get_residual_diffs(text)

        res = [inputs[:, None]]
        max_len = min(attns.shape[1], mlps.shape[1])
        for i in range(max_len):
            res.append(attns[:, i, None])
            res.append(mlps[:, i, None])
        res = torch.cat(res, dim=1)

        indices = torch.tensor([i-1 for i in data["indices"]])
        good_res = res[:, :, indices]
        batch.append(good_res)

    torch.save(batch, f"./tensors/res_data_{batch_index:03d}.pt")
# %%

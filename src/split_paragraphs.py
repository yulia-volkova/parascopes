# %%
from utils_load_data import load_embeds, load_paragraphs_with_context
from tqdm import tqdm

paragraphs = load_paragraphs_with_context()
split_paragraphs = {}

curr_idx = 0
for i in tqdm(range(100)):
    embeds = load_embeds(i)
    split_paragraphs[i] = paragraphs[curr_idx:curr_idx+len(embeds)]
    curr_idx += len(embeds)

print(len(embeds))
print(len(paragraphs))

# %%
import os
import json
os.makedirs("../data/split_paragraphs", exist_ok=True)

for i, split in split_paragraphs.items():
    with open(f"../data/split_paragraphs/paragraphs_with_context_{i:03d}.json", "w") as f:
        json.dump(split, f)

# %%

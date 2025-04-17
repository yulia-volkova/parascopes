# %%

# Get paragraphs with context
# I don't remeber what I was doing here
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

# Get the input paragraphs

import json
import os
from tqdm import tqdm
from utils_load_data import flatten, load_embeds

def load_paragraphs_from_file():
    file_path = "/workspace/SPAR/gen-dataset/new_split_dataset.jsonl"
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    splits = flatten([x["split"][:-1] for x in data])
    return splits

paragraphs = load_paragraphs_from_file()
split_paragraphs = {}

curr_idx = 0
for i in tqdm(range(100)):
    embeds = load_embeds(i)
    split_paragraphs[i] = paragraphs[curr_idx:curr_idx+len(embeds)]
    curr_idx += len(embeds)

print(len(embeds))
print(len(paragraphs))
print(curr_idx)
assert len(paragraphs) == curr_idx

os.makedirs("../data/split_input_paragraphs", exist_ok=True)
for i in range(100):
    with open(f"../data/split_input_paragraphs/paragraphs_{i:03d}.json", "w") as f:
        json.dump(split_paragraphs[i], f)


# %%
# convert split paragraphs into tokens
from utils_sonar import load_sonar_tokenizer
import json
from tqdm import tqdm

def load_tokenizer():
    return load_sonar_tokenizer("text_sonar_basic_encoder").create_encoder()
tokenizer = load_tokenizer()

for i in tqdm(range(100)):
    with open(f"../data/split_paragraphs/paragraphs_{i:03d}.json", "r") as f:
        paragraphs = json.load(f)
    tokens = []
    for paragraph in paragraphs:
        tokens.append(tokenizer(paragraph))
    # Use pickle instead of json for tokenized data as it may contain non-serializable objects
    import pickle
    with open(f"../data/split_paragraphs/paragraphs_tokenized_sonar_{i:03d}.pkl", "wb") as f:
        pickle.dump(tokens, f)

# %%



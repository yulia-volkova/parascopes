# %%
from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
torch.set_grad_enabled(False)

# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


DATASET_FILE = '/workspace/SPAR/gen-dataset/rp_outputs_3b_combined.jsonl'
with open(DATASET_FILE, 'r') as file:
    dataset = [json.loads(line) for line in file]
print(len(dataset))

format_prompt = lambda _text : f"""<|start_header_id|>user<|end_header_id|>
{_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# %%
lengths = []
for i, data in enumerate(dataset):
    lengths.append( (len(data["completion"]), i) )
    data["split"] = [ format_prompt(data["prompt"]), *data["completion"].split("\n\n") ]


print(max(lengths))
idx_max = max(lengths)[1]
#
for data in tqdm(dataset):
    full_text = data["split"][0]
    split_indices = [len(tokenizer.encode(full_text))]

    for text_piece in data["split"][1:-1]:
        full_text += text_piece + "\n\n"
        split_indices.append(len(tokenizer.encode(full_text)))

    data["indices"] = split_indices


# %%

OUTPUT_FILE = '/workspace/SPAR/gen-dataset/split_indexed_dataset.jsonl'
if os.path.exists(OUTPUT_FILE):
    raise Exception("File already exists")

for data in dataset:
    data["split"] = [
        data["split"][0],
        *[x+"\n\n" for x in data["split"][1:-1]],
        data["split"][-1]
    ]

with open(OUTPUT_FILE, 'w') as outfile:
    for data in dataset:
        del data['prompt']
        del data['completion']
        json.dump(data, outfile)
        outfile.write('\n')

# %%
# import json
# format_prompt = lambda _text : f"""<|start_header_id|>user<|end_header_id|>
# {_text}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """

# DATASET_FILE = '/workspace/SPAR/gen-dataset/rp_outputs_3b_combined.jsonl'
# OUTPUT_FILE = '/workspace/SPAR/gen-dataset/split_indexed_dataset.jsonl'
# NEW_OUTPUT = '/workspace/SPAR/gen-dataset/new_split_dataset.jsonl'
# with open(OUTPUT_FILE, 'r') as f:
#     dataset_new = [json.loads(line) for line in f]

# with open(DATASET_FILE, 'r') as f:
#     dataset_old = [json.loads(line) for line in f]

# for old, new in zip(dataset_old, dataset_new):
#     split = [format_prompt(old["prompt"]), *old["completion"].split("\n\n")]
#     split = [split[0], *[x+"\n\n" for x in split[1:-1]], split[-1]]
#     new["split"] = split

# with open(NEW_OUTPUT, 'w') as outfile:
#     for data in dataset_new:
#         json.dump(data, outfile)
#         outfile.write('\n')


# # %%

# %%
import json
import torch
import einops

BASE_DIR = "../data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
folder = BASE_DIR

def load_res_data_internal(index, group_size=2, group_operation="cat"):
    file_path = f"{folder}/res_tensors/res_data_{index:03d}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    data = torch.cat(data, dim=2)  # Concatenate list of tensors
    data = data.squeeze(0)
    data = einops.rearrange(data, 'layers samples dim -> samples layers dim')
    first_layer = data[:, 0:1, :]
    data = data[:, 1:, :]  # Remove first layer

    # Get cumulative activations data
    if "cum" in group_operation:
        data[:, 0, :] = data[:, 0, :] + first_layer.squeeze(1)
        data = torch.cumsum(data, dim=1)

    # split into groups
    data = einops.rearrange(data, 'samples (layersg g) dim -> samples layersg g dim', g=group_size)

    # apply operation to groups
    if group_operation == "sum":
        data = einops.reduce(data, 'samples layers g dim -> samples layers dim', reduction="sum")
        group_size = 1
    elif group_operation in ["cat", "cumcat"]:
        data = einops.rearrange(data, 'samples layers g dim -> samples layers (g dim)', g=group_size)
    else:
        raise ValueError(f"Unknown group operation: {group_operation}")

    # return for final processing
    return data

def load_res_data(index, group_size=2, groups_to_load=0, group_operation="cat"):
    data = load_res_data_internal(index, group_size, group_operation)
    data = einops.rearrange(data[:, -groups_to_load:], 'samples layersg gdim -> samples (layersg gdim)')
    return data.float()

def load_res_data_layer(index, layer_idx, group_size=2, group_operation="cat"):
    data = load_res_data_internal(index, group_size, group_operation)
    data = data[:, layer_idx, :]
    return data.float()

def load_embeds(index):
    file_path = f"{folder}/sonar_embeds/embeds_{index:03d}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    data = torch.cat(data, dim=0)
    return data.float()


from itertools import chain
def flatten(nested_list):
    return list(chain.from_iterable(nested_list))

def load_split_paragraphs(index):
    file_path = f"{folder}/split_paragraphs/paragraphs_{index:03d}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_split_paragraphs_tokenized(index):
    file_path = f"{folder}/split_paragraphs/paragraphs_tokenized_sonar_{index:03d}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_split_paragraphs_inputs(index):
    file_path = f"{folder}/split_input_paragraphs/paragraphs_{index:03d}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_paragraphs():
    file_path = "/workspace/SPAR/gen-dataset/new_split_dataset.jsonl"
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    splits = flatten([x["split"][1:] for x in data])
    return splits

def load_paragraphs_with_context(zipped=True):
    file_path = "/workspace/SPAR/gen-dataset/new_split_dataset.jsonl"
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    inputs  = flatten([x["split"][:-1] for x in data])
    outputs = flatten([x["split"][1:] for x in data])
    if zipped:
        return list(zip(inputs, outputs))
    else:
        return inputs, outputs

def load_full_contexts():
    file_path = "/workspace/SPAR/gen-dataset/new_split_dataset.jsonl"
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    contexts = []
    for item in data:
        tokens = item["split"][:-1]  # Exclude the final element
        # Build cumulative contexts by joining token prefixes with "".join()
        for i in range(1, len(tokens) + 1):
            context = "".join(tokens[:i])
            contexts.append(context)
    return contexts

if __name__ == "__main__":
    print(load_res_data(0).shape)
    print(load_embeds(0).shape)
    print(len(load_paragraphs()))
    print(len(load_split_paragraphs(0)))
    print(len(load_split_paragraphs_inputs(0)))

    assert load_split_paragraphs(0)[0] == load_split_paragraphs_inputs(0)[1]
# %%

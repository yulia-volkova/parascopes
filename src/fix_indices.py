# %%
import json

data_files = {
    "mlp": "processed_rubrics/ruberic_log_4o_best.jsonl_mlp.jsonl",
    "linear": "processed_rubrics/ruberic_log_4o_2.jsonl_linear.jsonl",
    "continued": "processed_rubrics/log4.txt_continued.jsonl",
    "baseline": "processed_rubrics/log4.txt_baseline.jsonl",
    "cheat-1": "processed_rubrics/log4.txt_cheat-1.jsonl",
    "cheat-5": "processed_rubrics/log4.txt_cheat-5.jsonl",
    "cheat-10": "processed_rubrics/log4.txt_cheat-10.jsonl",
    "regenerated": "processed_rubrics/log4.txt_regenerated.jsonl",
    "auto-decoded": "processed_rubrics/log4.txt_auto-decoded.jsonl"
}

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

data_lists = {}
data_dicts = {}

for method, file_path in data_files.items():
    data_lists[method] = load_jsonl(file_path)
    data_dicts[method] = {d["index"]: d for d in data_lists[method]}

get_indices = lambda key: set(data_dicts[key].keys())
get_intersecting_indices = lambda key: get_indices("mlp") & get_indices(key)
def test_reference_match(key):
    for index in get_intersecting_indices(key):
        assert data_dicts["mlp"][index]["reference"] == data_dicts[key][index]["reference"], (
            f"Reference mismatch for 'mlp' and '{key}' at index {index}"
            f"\nmlp: {json.dumps(data_dicts['mlp'][index]['reference'])}"
            f"\n{key}: {json.dumps(data_dicts[key][index]['reference'])}"
        )

print(len(get_indices("mlp")))
indices = list(get_indices("mlp") & get_indices("linear") & get_indices("continued") & get_indices("baseline") & get_indices("regenerated") & get_indices("auto-decoded"))
all_indices = list(get_indices("mlp") | get_indices("linear") | get_indices("continued") | get_indices("baseline") | get_indices("regenerated") | get_indices("auto-decoded"))
print(len(indices))
print(len(all_indices))

# get reference for all 10072 entries, first trying mlp, then linear, then
# continued, then baseline, then regenerated, then auto-decoded

references = {}
for index in all_indices:
    if index in data_dicts["mlp"]:
        references[index] = data_dicts["mlp"][index]["reference"]
    elif index in data_dicts["linear"]:
        references[index] = data_dicts["linear"][index]["reference"]
    elif index in data_dicts["continued"]:
        references[index] = data_dicts["continued"][index]["reference"]
    elif index in data_dicts["baseline"]:
        references[index] = data_dicts["baseline"][index]["reference"]
    elif index in data_dicts["regenerated"]:
        references[index] = data_dicts["regenerated"][index]["reference"]
    elif index in data_dicts["auto-decoded"]:
        references[index] = data_dicts["auto-decoded"][index]["reference"]
    else:
        raise ValueError(f"Index {index} not found in any dataset")

print(len(references))

test_reference_match("linear") # ok
test_reference_match("continued") # ok
test_reference_match("baseline") # ok
# test_reference_match("cheat-1") # broken
# test_reference_match("cheat-5") # broken
# test_reference_match("cheat-10") # broken
test_reference_match("regenerated") # ok
test_reference_match("auto-decoded") # ok


def return_fixed_index(orig_label, new_label):
    ref_idx = 0
    data_dicts[new_label] = {}
    for data in data_lists[orig_label]:
        new_data = data.copy()
        while new_data["reference"] != references[ref_idx]:
            ref_idx += 1
        assert new_data["reference"] == references[ref_idx]
        new_data["index"] = ref_idx
        data_dicts[new_label][ref_idx] = new_data
        ref_idx += 1
    return data_dicts[new_label]

return_fixed_index("cheat-1", "cheat-1-fixed")
return_fixed_index("cheat-5", "cheat-5-fixed")
return_fixed_index("cheat-10", "cheat-10-fixed")

test_reference_match("cheat-1-fixed")
test_reference_match("cheat-5-fixed")
test_reference_match("cheat-10-fixed")

data_dicts["cheat-1"] = data_dicts["cheat-1-fixed"]
data_dicts["cheat-5"] = data_dicts["cheat-5-fixed"]
data_dicts["cheat-10"] = data_dicts["cheat-10-fixed"]
del data_dicts["cheat-1-fixed"]
del data_dicts["cheat-5-fixed"]
del data_dicts["cheat-10-fixed"]

test_reference_match("cheat-1")
test_reference_match("cheat-5")
test_reference_match("cheat-10")

print("All references match")

print( len(get_indices("cheat-1")) )
print( len(get_indices("cheat-5")) )
print( len(get_indices("cheat-10")) )

with open("processed_rubrics/all_data_dicts.json", "w") as f:
    json.dump(data_dicts, f)

# %%

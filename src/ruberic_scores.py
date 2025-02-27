import os
import json
import ast  # Add ast module for literal evaluation
from collections import defaultdict

outputs = []
# file_name = "ruberic_log_4o_best.jsonl"
# file_name = "ruberic_log_4o_2.jsonl"
# file_name = "log4.txt"
file_name = "log5"
with open(file_name, "r", encoding="latin-1") as f:
    for line in list(f):
        try:
            data = ast.literal_eval(line)
            data['result'] = json.loads(data['result'])
            outputs.append(data)
        except:
            continue

outputs = list(sorted(outputs, key=lambda x: x['index']))

# Define the valid types based on your comparison texts
possible_types = [
    "continued", "baseline", "cheat-1", "cheat-5", "cheat-10",
    "regenerated", "auto-decoded", "mlp", "linear",
    "mlp-train", "linear-train",
]

# Convert old format to type field and group
grouped_outputs = defaultdict(list)
for item in outputs:
    # Find which type key exists in this item
    if 'type' not in item:
        for t in possible_types:
            if t in item:
                item['type'] = item.pop(t)  # Move value to 'type' and remove old key
                break
    # Add to appropriate group
    grouped_outputs[item['type']].append(item)
    # print(item['type'])

# Example usage:
for k, v in grouped_outputs.items():
    indices = [x["index"] for x in v]
    print(f"{k} Entries:", len(v), "Missing:", len(set(range(10072)) - set(indices)))

os.makedirs("processed_rubrics", exist_ok=True)
for k, v in grouped_outputs.items():
    with open(f"processed_rubrics/{file_name}_{k}.jsonl", "w") as f:
        for item in v:
            item['result'] = json.dumps(item['result'])
            f.write(json.dumps(item) + "\n")

# print(outputs[0])

# # Save the processed outputs back to a jsonl file
# with open("ruberic_scores/processed_jsonl_mlp.jsonl", "w") as f:
#     for output in outputs:
#         output['result'] = json.dumps(output['result'])
#         f.write(json.dumps(output) + "\n")



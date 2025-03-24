# %%

from utils_load_data import load_split_paragraphs
import json

paragraphs = load_split_paragraphs(0)
with open("../data/paragraphs_0_summaries.json", "r") as f:
    all_summaries = json.load(f)

Hs = ["Here is the formatted string:\n\n" in x for x in all_summaries]
for i, h in enumerate(Hs):
    if h:
        all_summaries[i] = all_summaries[i].replace("Here is the formatted string:\n\n", "")

L = len(all_summaries)

is_good      = ["A" == x[0] for x in all_summaries]
has_newline  = ["\n\n" in x for x in all_summaries]
has_brackets = ["[text type]" in x for x in all_summaries]


print(f"L_G: {sum(is_good)}, L_H: {sum(has_newline)}, L_B: {sum(has_brackets)}, L: {L}")

print(sum(is_good) + sum(has_newline) + sum(has_brackets))

non_matching = []
for i in range(len(all_summaries)):
    if not is_good[i] or has_newline[i] or has_brackets[i]:
        non_matching.append((all_summaries[i], paragraphs[i]))

print(len(non_matching), json.dumps(non_matching, indent=4))




# %%



# %%

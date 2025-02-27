import textwrap
from termcolor import colored
import argparse
import json
import torch
import einops
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace
from typing import List, Tuple

import utils_load_data
from taker import Model
from transformers import AutoTokenizer

torch.set_grad_enabled(False)

# %%
LIMIT = None

# DEFINE CODE FOR BATCHED GENERATION

def generate_batch_fast(m, input_ids, attn_masks, max_new_tokens, temperature, exclude_tokens=0) -> Tuple[List[str], List[str]]:
    if m.tokenizer.pad_token is not None:
        pass
    elif m.tokenizer.eos_token is not None:
        m.tokenizer.pad_token = m.tokenizer.eos_token
    else:
        raise ValueError("Tokenizer has neither pad_token nor eos_token defined.")

    orig_len = input_ids.shape[1]
    # Generate outputs
    generate_ids = m.predictor.generate(
        input_ids=input_ids.to(m.device),
        attention_mask=attn_masks.to(m.device),
        max_length=input_ids.shape[1] + max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=m.tokenizer.pad_token_id,
    )
    # Decode all generated sequences at once
    batch_text_after = m.tokenizer.batch_decode(
        [ids[orig_len:] for ids in generate_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Get original prompts
    batch_prompts = m.tokenizer.batch_decode(
        input_ids[:, exclude_tokens:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return batch_prompts, batch_text_after


def move_pad_tokens_to_left(token_ids, attn_mask, pad_token_id):
    # If there are any tokens that are m.tokenizer.pad_token_id, we need to swap
    # them to the left.
    # i.e: ids:  [a, b, c, d, 0, 0, 0] -> [0, 0, 0, a, b, c, d]
    # i.e: mask: [1, 1, 1, 1, 0, 0 ,0] -> [0, 0, 0, 1, 1, 1, 1]
    # Find where the padding tokens are
    pad_mask = token_ids == pad_token_id
    total_tokens = token_ids.shape[1]
    token_ids = token_ids.copy()
    attn_mask = attn_mask.copy()

    # For each row, count number of non-pad tokens and create new indices
    for i in range(len(token_ids)):
        non_pad_count = np.sum(~pad_mask[i])
        if non_pad_count < total_tokens:
            # Number of padding tokens needed
            n_pad = total_tokens - non_pad_count

            # Shift non-pad tokens to the right
            token_ids[i] = np.concatenate([
                np.full(n_pad, pad_token_id),
                token_ids[i][pad_mask[i] == 0]
            ])

            # Also shift attention mask
            attn_mask[i] = np.concatenate([
                np.zeros(n_pad),
                attn_mask[i][pad_mask[i] == 0]
            ])

    return token_ids, attn_mask


# %%
def main(verbose=False):
    args = SimpleNamespace(
        res_index=99,
        max_new_tokens=128,
        temperature=0.3
    )

    m = Model("meta-llama/Llama-3.2-3B-Instruct", dtype="bfp16", add_hooks=False)
    m.tokenizer.pad_token_id = m.tokenizer.eos_token_id
    m.show_details()


    # Load paragraphs
    embeds = utils_load_data.load_embeds(99)
    n_paragraphs = embeds.shape[0]
    contexts = utils_load_data.load_full_contexts()
    contexts = contexts[-n_paragraphs:]
    if LIMIT:
        contexts = contexts[:LIMIT]
    n_texts = len(contexts)

    # print(contexts[0])
    # print(contexts[1])
    # print(contexts[2])

    # Encode paragraphs
    context_tokens = m.tokenizer(contexts, padding=True, truncation=False, max_length=1000, return_tensors="pt")
    print(context_tokens.input_ids.shape, type(contexts), type(contexts[0]))
    orig_token_ids = context_tokens.input_ids
    orig_attn_mask = context_tokens.attention_mask
    orig_lengths = orig_attn_mask.sum(dim=1)

    sorted_indices = np.argsort(orig_lengths)

    attn_sorted = orig_attn_mask[sorted_indices]
    ids_sorted  = orig_token_ids[sorted_indices]
    lengths     = orig_lengths[sorted_indices]

    max_batch_tokens = 8000
    max_new_tokens = args.max_new_tokens

    i = 0
    batch_size = 16
    outputs = [""] * n_texts
    pbar = tqdm(total=n_texts, desc="Generating")
    while i < n_texts:
        # Determine the size of the current batch (in case fewer samples remain)
        j = i
        while True:
            j += 1
            if j >= n_texts:
                break
            if (j+1-i) * (lengths[j] + max_new_tokens) > max_batch_tokens:
                break

        curr_len = lengths[j-1]
        prompts = ids_sorted[i:j, -curr_len:].to(m.device)
        masks   = attn_sorted[i:j, -curr_len:].to(m.device)
        assert attn_sorted[i:j, :-curr_len].sum() == 0, f"Non-Padding tokens discarded, {attn_sorted[i:j, :-curr_len].shape} {attn_sorted[i:j, :-curr_len].sum()}, -{curr_len}"

        _prompts, texts = generate_batch_fast(m, prompts, masks, args.max_new_tokens, args.temperature, exclude_tokens=0)
        texts = [t.split('\n\n')[0] for t in texts]

        indices = sorted_indices[i:j]
        for text, index in zip(texts, indices):
            outputs[index] = text

        pbar.update(j-i)
        i = j


    pbar.close()

    result_data = {
        "model": m.model_repo,
        "prompt": "\n\n",
        "res_index": args.res_index,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "outputs": outputs,
    }
    out_filename = f"./comparison_texts/regenerated_outputs.json"
    with open(out_filename, "w") as f:
        f.write(json.dumps(result_data, indent=2) + "\n")
    print(f"Result written to {out_filename}")

if __name__ == "__main__":
    main()


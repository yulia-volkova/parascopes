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
N_CHEAT_TOKENS = 10
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
    inputs, outputs = utils_load_data.load_paragraphs_with_context(zipped=False)
    inputs, outputs = inputs[-n_paragraphs:], outputs[-n_paragraphs:]

    print(len(inputs), len(outputs), type(inputs), type(inputs[0]))

    # Encode paragraphs
    ids_inputs = m.tokenizer(inputs, padding=True, truncation=False, max_length=1000, return_tensors="pt")

    right_tokenizer = AutoTokenizer.from_pretrained(m.tokenizer_repo, legacy=False, padding_side='right')
    right_tokenizer.pad_token = m.tokenizer.pad_token
    ids_outputs = right_tokenizer(outputs, padding=True, truncation=False, max_length=1000, return_tensors="pt")

    # Transform into "cheat token" generation format
    n_tokens = 20
    n_texts = ids_inputs.input_ids.shape[0]
    inputs_lengths = ids_inputs.attention_mask.sum(axis=-1)
    outputs_lengths = ids_outputs.attention_mask.sum(axis=-1)

    # construct [bos, \n\n, tok1, tok2, ..., tok20]
    bos_tokens = ids_inputs.input_ids[np.arange(n_texts), -inputs_lengths][:, None]
    newline_tokens = ids_inputs.input_ids[:, -1][:, None]
    outputs_start_tokens = ids_outputs.input_ids[:, 1:n_tokens+1]
    token_counts = ids_outputs.attention_mask[:, 1:].sum(axis=-1)

    input_attentions = ids_inputs.attention_mask[:, -2:]
    output_attentions = ids_outputs.attention_mask[:, 1:n_tokens+1]

    print(ids_inputs.input_ids.shape, ids_outputs.input_ids.shape)

    if verbose:
        print("---")
        print("BOS   :", bos_tokens[:6, 0])
        print("\\n\\n  :", newline_tokens[:6, 0])
        print("toks..:", outputs_start_tokens[:6, 0])

    # [bos, \n\n, tok1, tok2, ..., tok20]
    token_ids = np.concatenate([bos_tokens, newline_tokens, outputs_start_tokens], axis=-1)
    attn_mask = np.concatenate([input_attentions, output_attentions], axis=-1)

    if verbose:
        print(token_ids.shape, attn_mask.shape)
        print(token_ids[0])
        print(attn_mask[0])
        for i in range(3):
            print(i, m.tokenizer.convert_ids_to_tokens(token_ids[i]))

    n_cheat_tokens = N_CHEAT_TOKENS
    total_tokens = 2 + n_cheat_tokens

    # fix offsets
    token_ids = token_ids[:, :total_tokens]
    attn_mask = attn_mask[:, :total_tokens]
    cheat_fracs = (attn_mask.sum(axis=-1) - 2) / token_counts
    token_ids, attn_mask = move_pad_tokens_to_left(token_ids, attn_mask, m.tokenizer.pad_token_id)

    if verbose:
        print(token_ids.shape, attn_mask.shape)
        print(m.tokenizer.convert_ids_to_tokens(token_ids[0]))


    batch_size = 20
    outputs = []
    for i in tqdm(range(0, n_texts, batch_size)):
        prompt = torch.tensor(token_ids[i:i+batch_size], dtype=torch.long, device=m.device)
        masks  = torch.tensor(attn_mask[i:i+batch_size], dtype=torch.long, device=m.device)
        prompts, texts = generate_batch_fast(m, prompt, masks, args.max_new_tokens, args.temperature, exclude_tokens=2)
        texts = [p+t for p, t in zip(prompts, texts)]
        texts = [t.split('\n\n')[0] for t in texts]
        outputs.extend(texts)

        if LIMIT and i+batch_size >= LIMIT:
            break

    result_data = {
        "model": m.model_repo,
        "cheat_tokens": n_cheat_tokens,
        "prompt": "\n\n",
        "res_index": args.res_index,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "outputs": outputs,
        "cheat_fracs": cheat_fracs.tolist(),
    }
    out_filename = f"./comparison_texts/baseline_{n_cheat_tokens}_outputs.json"
    with open(out_filename, "w") as f:
        f.write(json.dumps(result_data, indent=2) + "\n")
    print(f"Result written to {out_filename}")

if __name__ == "__main__":
    main()


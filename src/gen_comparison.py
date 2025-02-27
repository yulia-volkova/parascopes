import textwrap
from termcolor import colored
import argparse
import json
import torch
import einops
from tqdm import tqdm

import utils_load_data
from taker import Model

def main(verbose=False):
    parser = argparse.ArgumentParser(
        description="Generate transferred activations by directly replacing hook outputs with loaded diff segments."
    )
    parser.add_argument("--res_index", type=int, default=99,
                        help="The index used to load the res_data file (e.g. 99 loads tensors/res_data_099.pt)")
    # Use group_size=2 to obtain interleaved diff segments for [attn, mlp]
    parser.add_argument("--group_size", type=int, default=2,
                        help="Group size passed to load_res_data (should be 2 for interleaved [attn, mlp])")
    parser.add_argument("--groups_to_load", type=int, default=28,
                        help="The number of groups to load from res_data")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature")
    args = parser.parse_args()

    # Initialize the model (using your preferred model; here we use
    # Llama-3.2-3B-Instruct)
    hook_config = """
        post_attn: collect, replace
        post_mlp: collect, replace
    """

    m = Model("meta-llama/Llama-3.2-3B-Instruct", dtype="bfp16", hook_config=hook_config)
    m.tokenizer.pad_token_id = m.tokenizer.eos_token_id
    m.show_details()

    # To get device and token index info, obtain the neutral prompt's inputs embeddings.
    neutral_prompt = "\n\n"
    base_embeds = m.get_inputs_embeds(neutral_prompt)
    new_token_index = m.get_ids(neutral_prompt).shape[1] - 1
    device = base_embeds.device

    # Load the residual diff data (which are stored as diff values) and select one sample.
    transferred_diffs = utils_load_data.load_res_data(
        args.res_index,
        group_size=args.group_size,
        groups_to_load=args.groups_to_load
    )
    paragraphs = utils_load_data.load_paragraphs()[-transferred_diffs.shape[0]:]
    sample_diff = transferred_diffs.to(device)  # shape: [groups_to_load * (gdim)]
    print("Diffs shape", sample_diff.shape)

    # Determine the model's hidden size.
    d_model = m.cfg.d_model
    num_diff_segments = sample_diff.numel() // m.cfg.d_model
if num_diff_segments % 2 != 0:
        raise ValueError("Expected an even number of diff segments (for interleaved attn and mlp), got odd number.")

    # Reshape the flat diff into individual segments.
    diff_reshaped = sample_diff.view(-1, m.cfg.n_layers, 2, d_model)
    outputs = []

    for sample_idx, injected_diff in tqdm(list(enumerate(diff_reshaped))):
        # Reset all neuron replacement hooks.
        for hook in m.hooks.neuron_replace.values():
            hook.reset()

        # For each affected decoder block, directly replace the outputs of hook_attn and hook_mlp.
        for layer_index in range(m.cfg.n_layers):
            attn_injection = injected_diff[layer_index, 0]
            mlp_injection  = injected_diff[layer_index, 1]
            hook_attn = f"layer_{layer_index}_post_attn"
            hook_mlp  = f"layer_{layer_index}_post_mlp"
            m.hooks.neuron_replace[hook_attn].add_token(new_token_index, attn_injection)
            m.hooks.neuron_replace[hook_mlp].add_token(new_token_index, mlp_injection)
            # print(f"Replaced outputs of {hook_attn} and {hook_mlp} for token index {new_token_index}")

        # Generate new text with the model.
        output = m.generate(neutral_prompt, args.max_new_tokens, temperature=args.temperature)
        orig = paragraphs[sample_idx].split('\n')[0]
        gen = output[1].split('\n\n')[0]
        if verbose:
            print(textwrap.fill(colored(f"### {sample_idx} ###", "magenta"),
                                width=120,
                                initial_indent='',
                                subsequent_indent=' ' * 10))
            print(textwrap.fill(colored(f"ORIGINAL: {orig[:200]}", "blue"),
                                width=120,
                                initial_indent='',
                                subsequent_indent=' ' * 10))
            print(textwrap.fill(colored(f"GEN CONT: {gen[:200]}", "green"),
                                width=120,
                                initial_indent='',
                                subsequent_indent=' ' * 10))

        outputs.append(gen)

    result_data = {
        "model": m.model_repo,
        "prompt": neutral_prompt,
        "res_index": args.res_index,
        # "replaced_layers": replaced_layers,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "outputs": outputs,
    }
    out_filename = f"./transferred_activation_output.jsonl"
    with open(out_filename, "w") as f:
        f.write(json.dumps(result_data, indent=2) + "\n")
    print(f"Result written to {out_filename}")

if __name__ == "__main__":
    main()


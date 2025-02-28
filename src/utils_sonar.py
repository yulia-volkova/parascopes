# %%
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
import torch
import torch.nn.functional as F
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.data.text import TextTokenDecoder

def get_ce_loss_decoder(input_embed, expected_text, decoder="text_sonar_basic_decoder",
                       tokenizer="text_sonar_basic_encoder", target_lang="eng_Latn",
                       device="cpu", dtype=None):
    """
    Compute cross-entropy loss between decoder output and expected text.

    Args:
        input_embed: Tensor of shape [batch_size, embedding_dim] - SONAR embeddings
        expected_text: List of strings - the expected output text
        decoder: String or decoder model
        tokenizer: String or tokenizer
        target_lang: Target language code
        device: Device to run on
        dtype: Data type to use

    Returns:
        Cross-entropy loss
    """
    # Load the model
    model = EmbeddingToTextModelPipeline(
        decoder=decoder,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype
    )

    # Tokenize the expected text
    tokenizer_decoder = model.tokenizer.create_decoder()
    tokenizer_encoder = model.tokenizer.create_encoder(device=device)
    expected_tokens = [tokenizer_encoder(text) for text in expected_text]

    # Create padding mask and collate the expected tokens
    max_len = max(len(tokens) for tokens in expected_tokens)
    padded_tokens = []
    padding_mask_list = []

    for tokens in expected_tokens:
        pad_len = max_len - len(tokens)
        padded = torch.cat([tokens, torch.full((pad_len,), model.tokenizer.vocab_info.pad_idx,
                                              device=tokens.device, dtype=tokens.dtype)])
        padded_tokens.append(padded)
        padding_mask_list.append(torch.cat([torch.ones(len(tokens), dtype=torch.bool, device=tokens.device),
                                           torch.zeros(pad_len, dtype=torch.bool, device=tokens.device)]))

    target_tokens = torch.stack(padded_tokens).to(device)
    stacked_padding_mask = torch.stack(padding_mask_list)

    # Create PaddingMask with both the mask and sequence length
    seq_lens = torch.sum(stacked_padding_mask, dim=1).to(torch.int)
    padding_mask = PaddingMask(stacked_padding_mask, seq_lens)

    # Ensure input_embed is properly shaped [batch_size, 1, embedding_dim]
    if input_embed.dim() == 2:
        input_embed = input_embed.unsqueeze(1)

    # Forward pass through decoder model components
    with torch.no_grad():
        # Create a dummy batch for encoder output
        encoder_output = input_embed
        encoder_padding_mask = None

        # Get decoder frontend features
        target_tokens_shifted = target_tokens[:, :-1]  # Remove last token

        # Create a shifted padding mask - subtract 1 from sequence lengths to shift right
        shifted_seq_lens = torch.clamp(seq_lens - 1, min=0)
        target_padding_mask_shifted = PaddingMask(
            shifted_seq_lens,
            batch_seq_len=max_len - 1
        )

        seqs, padding_mask_dec = model.model.decoder.decoder_frontend(
            target_tokens_shifted, target_padding_mask_shifted)

        # Run through decoder
        seqs = seqs.to(device) if seqs is not None else None
        padding_mask_dec = padding_mask_dec.to(device) if padding_mask_dec is not None else None
        encoder_output = encoder_output.to(device) if encoder_output is not None else None
        encoder_padding_mask = encoder_padding_mask.to(device) if encoder_padding_mask is not None else None
        decoder_output, _ = model.model.decoder.decoder(
            seqs, padding_mask_dec, encoder_output, encoder_padding_mask)

        # Get logits by projecting
        logits = model.model.project(decoder_output, padding_mask_dec).logits

    # Calculate cross-entropy loss
    target_tokens_shifted = target_tokens[:, 1:]  # Remove first token (BOS)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_tokens_shifted.reshape(-1),
        ignore_index=model.tokenizer.vocab_info.pad_idx
    )

    # Calculate probabilities of correct tokens properly
    probs = F.softmax(logits, dim=-1)  # Apply softmax across vocab dimension

    # For each position, get the probability assigned to the correct token
    batch_size, seq_len, vocab_size = probs.shape
    target_probs = []

    for b in range(batch_size):
        for s in range(seq_len):
            if s < shifted_seq_lens[b]: # Only consider non-padding positions
                target_idx = target_tokens_shifted[b, s].item()
                if target_idx != model.tokenizer.vocab_info.pad_idx:
                    prob = probs[b, s, target_idx].item()
                    target_probs.append(prob)
                    # Avoid using decode directly on tokenizer_decoder
                    target_token = tokenizer_decoder(torch.tensor([target_idx], device=device))
                    print(f"Token ID {target_idx} {target_token} - Probability: {prob}")

                    # argmax of the probs
                    max_prob_idx = torch.argmax(probs[b, s])
                    max_token = tokenizer_decoder(torch.tensor([max_prob_idx], device=device))
                    print(f"Max ID {max_prob_idx} {max_token}: {probs[b, s, max_prob_idx].item()}")

    if target_probs:
        avg_prob = sum(target_probs) / len(target_probs)
        print(f"Average probability of correct tokens: {avg_prob}")

    return loss

if __name__ == "__main__":
    example_text = "Greetings from the Sonar team!"
    input_embed = torch.randn(1, 1, 1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = get_ce_loss_decoder(input_embed, [example_text], device=device)
    print(loss)
    text2vec = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device)
    vec2text = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=device)
    input_embed = text2vec.predict([example_text], source_lang="eng_Latn")
    output_text = vec2text.predict(input_embed, target_lang="eng_Latn")
    print(output_text)

    loss = get_ce_loss_decoder(input_embed, [example_text], device=device)
    print(loss)
# %%

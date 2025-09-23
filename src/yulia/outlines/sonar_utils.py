import torch
from typing import Optional, Tuple
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _default_dtype(dev: torch.device) -> torch.dtype:
    # Prefer bf16 on CUDA if supported; otherwise float32 everywhere.
    if dev.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float32

def init_sonar(
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    encoder_name: str = "text_sonar_basic_encoder",
    decoder_name: str = "text_sonar_basic_decoder",
    tokenizer_name: str = "text_sonar_basic_encoder",
) -> Tuple[TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline]:
    """
    Minimal initializer for SONAR encoder/decoder pipelines.
    Picks sensible defaults; falls back to CPU/float32 if init fails.
    """
    dev = device or _default_device()
    dt  = dtype  or _default_dtype(dev)
    print(f"[SONAR] init on device={dev}, dtype={dt}")

    try:
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        text2vec = TextToEmbeddingModelPipeline(
            encoder=encoder_name,
            tokenizer=tokenizer_name,
            device=dev,
            dtype=dt,
        )
        vec2text = EmbeddingToTextModelPipeline(
            decoder=decoder_name,
            tokenizer=tokenizer_name,
            device=dev,
            dtype=dt,
        )
        return text2vec, vec2text

    except Exception as e:
        print(f"[SONAR] init failed on {dev} ({dt}): {e} -> falling back to CPU/float32")
        cpu_dev, cpu_dt = torch.device("cpu"), torch.float32
        text2vec = TextToEmbeddingModelPipeline(
            encoder=encoder_name, tokenizer=tokenizer_name, device=cpu_dev, dtype=cpu_dt
        )
        vec2text = EmbeddingToTextModelPipeline(
            decoder=decoder_name, tokenizer=tokenizer_name, device=cpu_dev, dtype=cpu_dt
        )
        return text2vec, vec2text

import re
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer

"""
metrics.py

This module implements evaluation metrics for comparing generated outlines
against their source completions. The goal is to measure how well an outline
captures the main ideas of the completion while staying concise and avoiding redundancy.
"""


EMB = SentenceTransformer("all-MiniLM-L6-v2")
_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

def split_sentences(text: str) -> List[str]:
    if not text: return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return [s for s in sents if len(s.split()) >= 5]

def extract_outline_bullets(outline_md: str) -> List[str]:
    if not outline_md: return []
    lines = [ln.rstrip() for ln in outline_md.splitlines()]
    bullets = []
    for ln in lines:
        if re.match(r'^\s*\d+\.\s+', ln):
            bullets.append(re.sub(r'^\s*\d+\.\s+', '', ln).strip())
        elif re.match(r'^\s*[-•]\s+', ln):
            bullets.append(re.sub(r'^\s*[-•]\s+', '', ln).strip())
    return [b for b in bullets if b]

def embed_norm(texts: List[str]) -> np.ndarray:
    if not texts: return np.zeros((0, 384), dtype=np.float32)
    E = EMB.encode(texts, normalize_embeddings=True)
    return E

def positional_bias(n: int, lam: float = 0.12) -> np.ndarray:
    idx = np.arange(n)
    w = np.exp(-lam * idx)
    return w / (w.max() + 1e-12)

def centrality_scores(sent_embs: np.ndarray) -> np.ndarray:
    if len(sent_embs) == 0: return np.array([])
    centroid = sent_embs.mean(axis=0, keepdims=True)
    centroid /= np.linalg.norm(centroid) + 1e-12
    sims = (sent_embs @ centroid.T).ravel()
    return sims

def mmr_select(cand_embs: np.ndarray, doc_emb: np.ndarray, k: int, lambda_mm: float = 0.5) -> List[int]:
    if len(cand_embs) == 0 or k <= 0: return []
    k = min(k, len(cand_embs))
    selected = []
    sim_doc = cand_embs @ doc_emb
    sim_mat = cand_embs @ cand_embs.T
    while len(selected) < k:
        if not selected:
            i = int(np.argmax(sim_doc))
            selected.append(i)
            continue
        mask = np.ones(len(cand_embs), dtype=bool)
        mask[selected] = False
        max_sim_sel = sim_mat[:, selected].max(axis=1)
        mmr = lambda_mm * sim_doc + (1 - lambda_mm) * (-max_sim_sel)
        mmr[~mask] = -1e9
        j = int(np.argmax(mmr))
        selected.append(j)
    return selected


def pick_key_sentences(completion: str, top_k: int = 5, lead_bias: float = 0.35, lambda_mm: float = 0.5) -> Tuple[List[str], np.ndarray]:
    sents = split_sentences(completion)
    if not sents: return [], np.zeros((0, 384))
    S = embed_norm(sents)
    doc_emb = S.mean(axis=0, keepdims=True)
    central = centrality_scores(S)
    lead = positional_bias(len(S))
    scores = (1 - lead_bias) * central + lead_bias * lead
    order = np.argsort(-scores)
    cand_idx = order[: max(top_k * 4, top_k)]
    sel = mmr_select(S[cand_idx], doc_emb.ravel(), k=top_k, lambda_mm=lambda_mm)
    picked = [sents[cand_idx[i]] for i in sel]
    return picked, S[cand_idx][sel]


def semantic_prf1(completion: str, outline_md: str, top_k: int = 5, sim_threshold: float = 0.72, lead_bias: float = 0.35, lambda_mm: float = 0.5) -> Dict[str, float]:
    key_sents, key_embs = pick_key_sentences(completion, top_k=top_k, lead_bias=lead_bias, lambda_mm=lambda_mm)
    bullets = extract_outline_bullets(outline_md)
    if not key_sents or not bullets:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    B = embed_norm(bullets)
    S = key_embs @ B.T
    recall = float((S.max(axis=1) >= sim_threshold).mean())
    precision = float((S.max(axis=0) >= sim_threshold).mean())
    f1 = 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def conciseness_ratio(completion: str, outline_md: str) -> float:
    return len(outline_md) / max(1, len(completion))


def redundancy_score(outline_md: str) -> float:
    bullets = extract_outline_bullets(outline_md)
    if len(bullets) < 2: return 0.0
    E = embed_norm(bullets)
    S = E @ E.T
    iu = np.triu_indices(len(bullets), k=1)
    return float(S[iu].mean())

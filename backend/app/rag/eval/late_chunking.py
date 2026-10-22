"""Prototype de late chunking (phase 3, étape 3.6).

Découpage classique : chaque chunk est encodé seul, sans le texte qui l'entoure.
Late chunking : on encode le **document entier** en une passe (chaque token reçoit un
vecteur qui tient compte du contexte), puis on calcule le vecteur de chaque chunk à
partir des tokens de son segment.

Qwen3-Embedding est un modèle **causal** (un token ne voit que ce qui le précède) entraîné
à résumer un texte par son **dernier token**. Deux agrégations sont comparées :
- `mean` : moyenne des tokens du segment (méthode d'origine, pensée pour les modèles
  bidirectionnels) ;
- `last` : vecteur du dernier token du segment, qui a vu tout le texte précédent
  (cohérent avec l'entraînement de Qwen3).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.chunking import FRENCH_SEPARATORS, Chunk, TokenCounter, prepare_document
from app.rag.corpus import Article
from app.rag.eval.strategies import build

Pooling = Literal["mean", "last"]


def spans(article: Article, count: TokenCounter, size: int = 256, overlap: int = 25):
    """Chunks récursifs avec leur position (début, fin) en caractères dans le document."""
    document = prepare_document(article)
    splitter = RecursiveCharacterTextSplitter(
        separators=FRENCH_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=count,
        add_start_index=True,
    )
    pieces = splitter.create_documents([document])
    return document, [(p.metadata["start_index"], p.page_content) for p in pieces]


def late_chunk_vectors(
    model: Any, document: str, pieces: list[tuple[int, str]], pooling: Pooling
) -> np.ndarray:
    """Encode le document en une passe, puis agrège les tokens de chaque segment."""
    import torch

    tokenizer = model.tokenizer
    encoded = tokenizer(document, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    transformer = model[0].auto_model
    device = next(transformer.parameters()).device
    with torch.no_grad():
        hidden = transformer(**{k: v.to(device) for k, v in encoded.items()}).last_hidden_state[0]

    vectors = []
    for start, text in pieces:
        end = start + len(text)
        tokens = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]
        segment = hidden[tokens[0] : tokens[-1] + 1]
        vector = segment.mean(dim=0) if pooling == "mean" else segment[-1]
        vectors.append(torch.nn.functional.normalize(vector, dim=0).float().cpu().numpy())
    return np.stack(vectors)


def late_chunks(
    article: Article, count: TokenCounter, model: Any, pooling: Pooling
) -> tuple[list[Chunk], np.ndarray]:
    document, pieces = spans(article, count)
    chunks = build(article, [("", text) for _, text in pieces], f"late-{pooling}", count)
    return chunks, late_chunk_vectors(model, document, pieces, pooling)

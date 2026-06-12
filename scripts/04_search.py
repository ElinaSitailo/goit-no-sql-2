"""Семантичний пошук у Pinecone та локальне порівняння метрик."""

import os
from typing import Any

import numpy as np
import pandas as pd
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
DATA_FILE = "data/arxiv_subset.parquet"
EMBEDDINGS_FILE = "embeddings/embeddings.npy"
TOP_K = 5


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)

def get_pinecone_index(index_name: str):
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        raise ValueError("Не знайдено PINECONE_API_KEY у змінних оточення.")

    pinecone_class = getattr(pinecone, "Pinecone", None)
    if pinecone_class is not None:
        pc = pinecone_class(api_key=api_key)
        return pc.Index(index_name)

    if hasattr(pinecone, "init"):
        pinecone.init(api_key=api_key)
        index_factory = getattr(pinecone, "Index", None)
        if index_factory is None:
            raise RuntimeError("У SDK Pinecone відсутній конструктор Index.")
        return index_factory(index_name)

    raise RuntimeError("Не вдалося ініціалізувати Pinecone-клієнт у поточному SDK.")


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embedding[0]


def print_pinecone_results(title: str, results: Any) -> None:
    print(f"\n{title}")
    matches = getattr(results, "matches", None)
    if matches is None and isinstance(results, dict):
        matches = results.get("matches", [])
    matches = matches or []

    for rank, match in enumerate(matches, start=1):
        metadata = getattr(match, "metadata", None)
        score = getattr(match, "score", None)
        if isinstance(match, dict):
            metadata = match.get("metadata", {})
            score = match.get("score", 0.0)

        metadata = metadata or {}
        abstract = _to_text(metadata.get("abstract", ""))
        safe_score = float(score) if score is not None else 0.0
        print(
            f"{rank}. score={safe_score:.4f} | {metadata.get('title', 'N/A')}\n"
            f"   category={metadata.get('category', 'N/A')} | year={metadata.get('year', 'N/A')}\n"
            f"   abstract: {abstract[:200]}..."
        )


def local_top_k(df: pd.DataFrame, scores: np.ndarray, metric_name: str, top_k: int = TOP_K) -> None:
    order = np.argsort(-scores)[:top_k]
    print(f"\nЛокальний топ-{top_k} для {metric_name}:")
    for rank, idx in enumerate(order, start=1):
        row = df.iloc[int(idx)]
        abstract = _to_text(row.get("abstract", ""))
        print(
            f"{rank}. score={float(scores[idx]):.4f} | {row.get('title', 'N/A')}\n"
            f"   category={row.get('category', 'N/A')} | year={row.get('year', 'N/A')}\n"
            f"   abstract: {abstract[:200]}..."
        )


def main() -> None:
    query = "teaching machines to recognize objects in pictures"
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = encode_query(model, query)

    index = get_pinecone_index(INDEX_NAME)

    pure_results = index.query(
        vector=query_embedding.tolist(),
        top_k=TOP_K,
        include_metadata=True,
    )
    print_pinecone_results("Чистий семантичний пошук (top-5):", pure_results)

    recent_year = pd.Timestamp.now().year - 5
    filter_a = {
        "$and": [
            {"category": {"$eq": "cs.LG"}},
            {"year": {"$gte": recent_year}},
        ]
    }
    rl_query = "reinforcement learning"
    rl_embedding = encode_query(model, rl_query)
    filtered_a = index.query(
        vector=rl_embedding.tolist(),
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_a,
    )
    print_pinecone_results("Фільтр A: RL, останні 5 років, category=cs.LG", filtered_a)

    filter_b = {"year": {"$lte": 2015}}
    filtered_b = index.query(
        vector=query_embedding.tolist(),
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_b,
    )
    print_pinecone_results("Фільтр B: старіші статті (year <= 2015)", filtered_b)

    print(
        "\nПояснення різниці видачі:\n"
        "- У фільтрі A видача звужена до сучасних робіт cs.LG, тому результати більш спеціалізовані.\n"
        "- У фільтрі B видані старіші статті: вони можуть бути менш точними щодо сучасної термінології, "
        "але краще показують базові/класичні підходи."
    )

    df = pd.read_parquet(DATA_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    if len(df) != len(embeddings):
        raise ValueError("Кількість записів у parquet і embeddings.npy не збігається.")

    cosine_scores = embeddings @ query_embedding
    dot_scores = embeddings @ query_embedding
    l2_scores = -np.linalg.norm(embeddings - query_embedding, axis=1)

    local_top_k(df, cosine_scores, "cosine similarity")
    local_top_k(df, dot_scores, "dot product")
    local_top_k(df, l2_scores, "L2-distance (через -distance)")


if __name__ == "__main__":
    main()
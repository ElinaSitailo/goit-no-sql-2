"""Завантаження ембеддингів arXiv у Pinecone."""

import os
from typing import Any

import numpy as np
import pandas as pd
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


INDEX_NAME = "arxiv-papers"
DATA_FILE = "data/arxiv_subset.parquet"
EMBEDDINGS_FILE = "embeddings/embeddings.npy"
BATCH_SIZE = 200


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _truncate(value: Any, max_len: int) -> str:
    return _to_str(value)[:max_len]


def _extract_index_names(indexes: Any) -> list[str]:
    if hasattr(indexes, "names"):
        return list(indexes.names())
    if isinstance(indexes, dict) and "indexes" in indexes:
        return [item.get("name") for item in indexes["indexes"] if item.get("name")]
    if hasattr(indexes, "indexes"):
        names = []
        for item in indexes.indexes:
            name = getattr(item, "name", None)
            if name:
                names.append(name)
        return names
    return []


def create_index_if_not_found(dimension: int, pc) -> pinecone.Index:
    existing_indexes = _extract_index_names(pc.list_indexes())
    # створення індексу, якщо він ще не існує
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(INDEX_NAME)


def main() -> None:
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        raise ValueError("Не знайдено PINECONE_API_KEY у змінних оточення.")

    api_key = api_key.strip().strip('"').strip("'")

    df = pd.read_parquet(DATA_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Кількість записів у датасеті ({len(df)}) не збігається з кількістю ембеддингів ({len(embeddings)})."
        )

    if embeddings.ndim != 2:
        raise ValueError("Очікується 2D-масив ембеддингів.")

    dimension = int(embeddings.shape[1])

    pc = Pinecone(api_key=api_key)

    index = create_index_if_not_found(dimension, pc)


    total = len(df)
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Завантаження у Pinecone"):
        end = min(start + BATCH_SIZE, total)
        batch_vectors = []

        for row_idx in range(start, end):
            row = df.iloc[row_idx]
            batch_vectors.append(
                {
                    "id": f"paper_{row_idx}",
                    "values": embeddings[row_idx].tolist(),
                    "metadata": {
                        "arxiv_id": _to_str(row.get("id", "")),
                        "title": _to_str(row.get("title", "")),

                        # abstract обрізається до 500 символів, оскільки Pinecone обмежує сумарний розмір метаданих одного вектора до 40 KB.
                        # Повний текст анотації потрібно зберігати окремо (у parquet-файлі) і підтягувати за ID після пошуку.
                        "abstract": _truncate(row.get("abstract", ""), 500),
                        "authors": _truncate(row.get("authors", ""), 200),
                        "year": int(row.get("year", 0)) if not pd.isna(row.get("year", 0)) else 0,
                        "category": _to_str(row.get("category", "")),
                    },
                }
            )

        index.upsert(vectors=batch_vectors)

    # Статистика індексу
    stats = index.describe_index_stats()
    if isinstance(stats, dict):
        total_vectors = stats.get("total_vector_count", 0)
    else:
        total_vectors = getattr(stats, "total_vector_count", 0)

    print(f"Загальна кількість векторів в індексі: {total_vectors}")


if __name__ == "__main__":
    main()
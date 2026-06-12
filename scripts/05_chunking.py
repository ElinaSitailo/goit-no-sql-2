"""Chunking анотацій arXiv та завантаження чанків у Pinecone."""

import os
import re
from typing import Any, cast

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_NAME = "allenai/specter2_base"
DATA_FILE = "data/arxiv_subset.parquet"
FIXED_INDEX_NAME = "arxiv-chunks-fixed"
SEMANTIC_INDEX_NAME = "arxiv-chunks-semantic"

TOP_N_LONGEST = 30
FIXED_CHUNK_WORDS = 120
FIXED_OVERLAP_WORDS = 20
SEMANTIC_MAX_WORDS = 120
UPSERT_BATCH_SIZE = 100
TOP_K = 5

TEST_QUERIES = [
    "transformers for long text understanding",
    "reinforcement learning with neural networks",
    "graph neural networks for molecular property prediction",
]


def _to_text(value: Any) -> str:
    """Turn a value into a text string."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)

def split_fixed(text: str, chunk_words: int = FIXED_CHUNK_WORDS, overlap_words: int = FIXED_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def split_semantic(text: str, max_words: int = SEMANTIC_MAX_WORDS) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)

        if sentence_len > max_words:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_words = 0
            for part in split_fixed(sentence, chunk_words=max_words, overlap_words=0):
                chunks.append(part)
            continue

        if current_words + sentence_len <= max_words:
            current_sentences.append(sentence)
            current_words += sentence_len
        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_words = sentence_len

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def build_chunk_records(df: pd.DataFrame, strategy: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        row_idx_int = int(cast(int, row_idx))
        abstract = _to_text(row.get("abstract", "")).strip()
        if not abstract:
            continue

        if strategy == "fixed":
            chunk_texts = split_fixed(abstract)
        else:
            chunk_texts = split_semantic(abstract)

        for chunk_idx, chunk_text in enumerate(chunk_texts):
            records.append(
                {
                    "id": f"{strategy}_{row_idx_int}_{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        "arxiv_id": _to_text(row.get("id", "")),
                        "title": _to_text(row.get("title", "")),
                        "chunk_text": chunk_text[:900],
                        "chunk_idx": int(chunk_idx),
                        "year": int(row.get("year", 0)) if not pd.isna(row.get("year", 0)) else 0,
                        "category": _to_text(row.get("category", "")),
                    },
                }
            )

    return records


def create_index_if_missing(pc: Pinecone, index_name: str, dimension: int) -> None:
    existing = pc.list_indexes()
    existing_names = list(existing.names()) if hasattr(existing, "names") else []
    if index_name in existing_names:
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def upsert_chunks(index, model: SentenceTransformer, chunk_records: list[dict[str, Any]], label: str) -> None:
    for start in tqdm(range(0, len(chunk_records), UPSERT_BATCH_SIZE), desc=f"Upsert {label}"):
        end = min(start + UPSERT_BATCH_SIZE, len(chunk_records))
        batch = chunk_records[start:end]
        texts = [item["text"] for item in batch]

        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        vectors = []
        for i, item in enumerate(batch):
            vectors.append(
                {
                    "id": item["id"],
                    "values": embeddings[i].tolist(),
                    "metadata": item["metadata"],
                }
            )

        index.upsert(vectors=vectors)


def search_chunks(index, model: SentenceTransformer, query: str, label: str) -> None:
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    result = index.query(
        vector=query_embedding.tolist(),
        top_k=TOP_K,
        include_metadata=True,
    )

    print(f"\n{label} | Запит: {query}")
    matches = getattr(result, "matches", []) or []
    for rank, match in enumerate(matches, start=1):
        metadata = getattr(match, "metadata", {}) or {}
        score = getattr(match, "score", 0.0)
        chunk_text = _to_text(metadata.get("chunk_text", ""))
        print(
            f"{rank}. score={float(score):.4f} | {metadata.get('title', 'N/A')}\n"
            f"   category={metadata.get('category', 'N/A')} | year={metadata.get('year', 'N/A')} | "
            f"chunk_idx={metadata.get('chunk_idx', 'N/A')}\n"
            f"   chunk: {chunk_text[:220]}..."
        )


def main() -> None:
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        raise ValueError("Не знайдено PINECONE_API_KEY у змінних оточення.")

    df = pd.read_parquet(DATA_FILE)
    df = df.copy()
    df["abstract_len"] = df["abstract"].fillna("").astype(str).str.split().str.len()
    top_df = df.sort_values("abstract_len", ascending=False).head(TOP_N_LONGEST)

    print(f"Відібрано {len(top_df)} статей із найдовшими анотаціями.")

    fixed_chunks = build_chunk_records(top_df, strategy="fixed")
    semantic_chunks = build_chunk_records(top_df, strategy="semantic")

    print(f"Fixed chunks: {len(fixed_chunks)}")
    print(f"Semantic chunks: {len(semantic_chunks)}")

    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_embedding_dimension()
    dimension = int(embedding_dim if embedding_dim is not None else 768)

    pc = Pinecone(api_key=api_key)
    create_index_if_missing(pc, FIXED_INDEX_NAME, dimension)
    create_index_if_missing(pc, SEMANTIC_INDEX_NAME, dimension)

    fixed_index = pc.Index(FIXED_INDEX_NAME)
    semantic_index = pc.Index(SEMANTIC_INDEX_NAME)

    upsert_chunks(fixed_index, model, fixed_chunks, label="fixed")
    upsert_chunks(semantic_index, model, semantic_chunks, label="semantic")

    fixed_stats = fixed_index.describe_index_stats()
    semantic_stats = semantic_index.describe_index_stats()
    fixed_total = getattr(fixed_stats, "total_vector_count", 0)
    semantic_total = getattr(semantic_stats, "total_vector_count", 0)
    print(f"\nВекторів у {FIXED_INDEX_NAME}: {fixed_total}")
    print(f"Векторів у {SEMANTIC_INDEX_NAME}: {semantic_total}")

    for query in TEST_QUERIES:
        search_chunks(fixed_index, model, query, label="Fixed-size")
        search_chunks(semantic_index, model, query, label="Semantic")


if __name__ == "__main__":
    main()
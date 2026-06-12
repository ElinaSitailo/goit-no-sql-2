"""Гібридний пошук: BM25 + Pinecone (vector) через Reciprocal Rank Fusion."""

import os
import re
from typing import Any

import pandas as pd
import pinecone
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
DATA_FILE = "data/arxiv_subset.parquet"
TOP_K = 5
RRF_K = 60
RETRIEVE_K = 50

TEST_QUERIES = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _build_doc_text(row: pd.Series) -> str:
    title = _to_text(row.get("title", "")).strip()
    abstract = _to_text(row.get("abstract", "")).strip()
    return f"{title} {abstract}".strip()


def get_pinecone_index(index_name: str):
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        raise ValueError("Не знайдено PINECONE_API_KEY у змінних оточення.")

    api_key = api_key.strip().strip('"').strip("'")

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


def encode_query(model: SentenceTransformer, query: str) -> list[float]:
    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return vector.tolist()


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    df: pd.DataFrame,
    tokenized_docs: list[list[str]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    valid_indices = [i for i, tokens in enumerate(tokenized_docs) if tokens]
    sorted_indices = sorted(valid_indices, key=lambda i: float(scores[i]), reverse=True)

    results: list[dict[str, Any]] = []
    for idx in sorted_indices[:top_k]:
        row = df.iloc[idx]
        results.append(
            {
                "id": f"paper_{idx}",
                "score": float(scores[idx]),
                "title": _to_text(row.get("title", "N/A")),
                "category": _to_text(row.get("category", "N/A")),
                "year": int(row.get("year", 0)) if not pd.isna(row.get("year", 0)) else 0,
                "abstract": _to_text(row.get("abstract", "")),
            }
        )

    return results


def vector_search(index, model: SentenceTransformer, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    query_vector = encode_query(model, query)
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    matches = getattr(response, "matches", None)
    if matches is None and isinstance(response, dict):
        matches = response.get("matches", [])

    results: list[dict[str, Any]] = []
    for match in matches or []:
        doc_id = getattr(match, "id", None)
        metadata = getattr(match, "metadata", None)
        score = getattr(match, "score", None)

        if isinstance(match, dict):
            doc_id = match.get("id")
            metadata = match.get("metadata", {})
            score = match.get("score", 0.0)

        metadata = metadata or {}
        results.append(
            {
                "id": _to_text(doc_id),
                "score": float(score) if score is not None else 0.0,
                "title": _to_text(metadata.get("title", "N/A")),
                "category": _to_text(metadata.get("category", "N/A")),
                "year": metadata.get("year", "N/A"),
                "abstract": _to_text(metadata.get("abstract", "")),
            }
        )

    return results


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    df: pd.DataFrame,
    top_k: int = TOP_K,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(bm25_results, start=1):
        doc_id = _to_text(item.get("id", ""))
        if not doc_id:
            continue
        fused.setdefault(doc_id, {"id": doc_id, "rrf_score": 0.0})
        fused[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, item in enumerate(vector_results, start=1):
        doc_id = _to_text(item.get("id", ""))
        if not doc_id:
            continue
        fused.setdefault(doc_id, {"id": doc_id, "rrf_score": 0.0})
        fused[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda x: float(x["rrf_score"]), reverse=True)[:top_k]

    enriched: list[dict[str, Any]] = []
    for item in ranked:
        doc_id = _to_text(item["id"])
        idx_text = doc_id.replace("paper_", "")
        if idx_text.isdigit():
            row = df.iloc[int(idx_text)]
            enriched.append(
                {
                    "id": doc_id,
                    "rrf_score": float(item["rrf_score"]),
                    "title": _to_text(row.get("title", "N/A")),
                    "category": _to_text(row.get("category", "N/A")),
                    "year": int(row.get("year", 0)) if not pd.isna(row.get("year", 0)) else 0,
                    "abstract": _to_text(row.get("abstract", "")),
                }
            )
        else:
            enriched.append(
                {
                    "id": doc_id,
                    "rrf_score": float(item["rrf_score"]),
                    "title": "N/A",
                    "category": "N/A",
                    "year": "N/A",
                    "abstract": "",
                }
            )

    return enriched


def print_results(title: str, results: list[dict[str, Any]], score_key: str) -> None:
    print(f"\n{title}")
    for rank, item in enumerate(results, start=1):
        abstract = _to_text(item.get("abstract", ""))
        score = float(item.get(score_key, 0.0))
        print(
            f"{rank}. {score_key}={score:.4f} | {item.get('title', 'N/A')}\n"
            f"   category={item.get('category', 'N/A')} | year={item.get('year', 'N/A')}\n"
            f"   abstract: {abstract[:220]}..."
        )


def main() -> None:
    df = pd.read_parquet(DATA_FILE)
    documents = [_build_doc_text(row) for _, row in df.iterrows()]
    tokenized_docs = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    model = SentenceTransformer(MODEL_NAME)
    index = get_pinecone_index(INDEX_NAME)

    print(f"Побудовано BM25-індекс для {len(df)} документів.")

    for query in TEST_QUERIES:
        print(f"\n{'=' * 90}\nЗапит: {query}")

        bm25_results = bm25_search(query, bm25, df, tokenized_docs, top_k=TOP_K)
        vector_results = vector_search(index, model, query, top_k=RETRIEVE_K)
        hybrid_results = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            df=df,
            top_k=TOP_K,
            rrf_k=RRF_K,
        )

        print_results("BM25 top-5:", bm25_results, score_key="score")
        print_results("Vector (Pinecone) top-5:", vector_results[:TOP_K], score_key="score")
        print_results("Hybrid (RRF) top-5:", hybrid_results, score_key="rrf_score")


if __name__ == "__main__":
    main()
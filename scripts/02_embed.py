"""Генерація ембеддингів для підмножини arXiv за допомогою SPECTER2."""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = Path("embeddings")
OUTPUT_FILE = OUTPUT_DIR / "embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

def main() -> None:

    # завантаження `data/arxiv_subset.parquet` через `pandas`
    df = pd.read_parquet(INPUT_FILE)

    # формуємо вхідні тексти
    texts = (
        df["title"].fillna("").astype(str)
        + " [SEP] "
        + df["abstract"].fillna("").astype(str)
    ).tolist()

    # формуємо ембеддинги пакетно з прогресом і нормалізацією
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    first_norm = float(np.linalg.norm(embeddings[0])) if len(embeddings) > 0 else 0.0

    print(f"Кількість оброблених текстів: {len(texts)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1] if embeddings.ndim == 2 else 0}")
    print(f"Норма першого ембеддингу: {first_norm:.6f}")

    # збереження ембеддингів у файл в `embeddings` директорію
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)
    print(f"Ембеддинги збережено у: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
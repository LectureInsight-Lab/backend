"""
KR-SBERT 임베딩 유틸리티 (항목 3-3, 4-2 공용).

첫 호출 시 모델을 로드하고 lru_cache로 재사용.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

_KR_SBERT_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    return SentenceTransformer(_KR_SBERT_MODEL)


def embed(texts: list[str]) -> NDArray[np.float32]:
    """텍스트 리스트를 KR-SBERT로 인코딩. 단위 벡터로 정규화해 반환."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_sim(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """정규화된 벡터 쌍의 코사인 유사도 (내적과 동일)."""
    return float(np.dot(a, b))

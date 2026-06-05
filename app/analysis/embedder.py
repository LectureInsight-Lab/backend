"""RAG 인덱싱 + 검색 (2-B단계) — Papers #1, #5.

강의 전체를 15행 단위 청크로 분할 → text-embedding-3-small 로 임베딩
→ 각 체크리스트 항목의 평가 기준 문장을 쿼리로 코사인 유사도 검색
→ top-K 청크를 LLM 컨텍스트로 주입.

기존 v1의 단순 "구간 고정 추출" 또는 "키워드 매칭" 을 의미 기반 검색으로 대체.

캐싱:
- 동일 강의 재분석 시 임베딩 재호출 비용을 막기 위해
  ``data/processed/embeddings/{lecture_date}.json`` 에 저장.

TODO:
- OpenAI Embedding API 래퍼 (text-embedding-3-small)
- 코사인 유사도 검색 (numpy / scikit-learn)
- 캐시 hit/miss 처리
- 타임스탬프 보존 (청크 경계에 시작/종료 시각 부착)
"""
from app.analysis.schemas import IndexedChunk, LectureDocument, LectureIndex


def build_index(document: LectureDocument, chunk_lines: int = 15) -> LectureIndex:
    """LectureDocument → 청크 분할 → 임베딩 → LectureIndex (placeholder)."""
    raise NotImplementedError


def embed_texts(texts: list[str]) -> list[list[float]]:
    """OpenAI Embedding 호출 (캐싱 포함, placeholder)."""
    raise NotImplementedError


def search(
    index: LectureIndex,
    query: str,
    top_k: int = 5,
) -> list[IndexedChunk]:
    """쿼리에 의미적으로 가장 가까운 청크 top-K 반환 (placeholder)."""
    raise NotImplementedError

"""RAG 인덱싱 + 검색 (2-B단계) — Papers #1, #5.

강의 전체를 ``rag_chunk_lines`` 행 단위 청크로 분할하고, 각 체크리스트 항목의
평가 기준 문장을 쿼리로 **키워드(토큰) 오버랩** 검색해 top-K 청크를 LLM 컨텍스트로 준다.

설계 결정: 임베딩 없이 키워드 검색만 사용한다(외부 임베딩 API 호출 없음).
- 한국어는 어절이 굴절되므로, 쿼리 토큰(길이 ≥ 2)이 청크 텍스트에 부분 문자열로
  등장하는지로 매칭한다(형태소 분석 불필요, STT 강건).
- 점수 = 매칭된 고유 쿼리 토큰 수 (+ 총 등장 횟수로 미세 가중). 동점은 chunk_id 오름차순.
"""
from __future__ import annotations

import re

from app.analysis.schemas import IndexedChunk, LectureDocument, LectureIndex
from app.core.config import settings

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_MIN_TOKEN_LEN = 2


def _tokenize(text: str) -> list[str]:
    """영문/숫자/한글 토큰 추출 (길이 1 토큰 제외, 소문자화)."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= _MIN_TOKEN_LEN]


def build_index(document: LectureDocument, chunk_lines: int | None = None) -> LectureIndex:
    """LectureDocument → ``chunk_lines`` 행 단위 청크 분할 → LectureIndex.

    임베딩은 계산하지 않는다(``embedding=None``). 검색은 ``search()`` 의 키워드 매칭.
    """
    chunk_lines = chunk_lines or settings.rag_chunk_lines
    lines = document.all_lines
    chunks: list[IndexedChunk] = []

    for chunk_id, start in enumerate(range(0, len(lines), chunk_lines)):
        window = lines[start : start + chunk_lines]
        if not window:
            continue
        chunks.append(
            IndexedChunk(
                chunk_id=chunk_id,
                start_timestamp=window[0].timestamp,
                end_timestamp=window[-1].timestamp,
                text=" ".join(u.text for u in window),
                embedding=None,
                line_indices=list(range(start, start + len(window))),
            )
        )

    return LectureIndex(lecture_date=document.lecture_date, chunks=chunks)


def _score(query_tokens: set[str], chunk_text: str) -> float:
    """쿼리 토큰과 청크 텍스트의 오버랩 점수.

    매칭된 고유 토큰 수가 1차 기준, 총 등장 횟수가 미세 가중(0.01)으로 동점 분리.
    """
    if not query_tokens:
        return 0.0
    lowered = chunk_text.lower()
    matched = 0
    occurrences = 0
    for tok in query_tokens:
        c = lowered.count(tok)
        if c:
            matched += 1
            occurrences += c
    return matched + 0.01 * occurrences


def search(index: LectureIndex, query: str, top_k: int | None = None) -> list[IndexedChunk]:
    """쿼리에 키워드가 가장 많이 겹치는 청크 top-K 반환.

    매칭이 전혀 없으면(점수 0) 폴백으로 앞쪽 청크 top-K 를 반환해 컨텍스트 공백을 막는다.
    """
    top_k = top_k or settings.rag_top_k
    if not index.chunks:
        return []

    query_tokens = set(_tokenize(query))
    scored = [(_score(query_tokens, c.text), c) for c in index.chunks]
    # 동점은 chunk_id 오름차순 (앞 구간 우선)
    scored.sort(key=lambda sc: (-sc[0], sc[1].chunk_id))

    if scored[0][0] == 0.0:                       # 매칭 전무 → 앞쪽 청크 폴백
        return index.chunks[:top_k]
    return [c for score, c in scored[:top_k] if score > 0.0][:top_k]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """임베딩은 비활성화됨 (키워드 검색 채택). 호출 시 명시적 에러."""
    raise NotImplementedError(
        "임베딩은 비활성화되어 있습니다. RAG 는 search() 의 키워드 검색을 사용하세요."
    )

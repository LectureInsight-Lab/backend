"""tests/test_embedder.py — 키워드 기반 RAG 인덱싱/검색 (임베딩 없음)."""
from __future__ import annotations

from app.analysis import embedder
from app.analysis.schemas import LectureDocument, Utterance


def _doc(texts: list[str]) -> LectureDocument:
    lines = [
        Utterance(timestamp="09:00:00", speaker_id="a", text=t, seconds_from_start=i * 5)
        for i, t in enumerate(texts)
    ]
    return LectureDocument(lecture_date="2026-02-02", instructor_id="i", all_lines=lines)


def test_build_index_chunking():
    doc = _doc([f"문장{i}" for i in range(35)])
    index = embedder.build_index(doc, chunk_lines=15)
    assert len(index.chunks) == 3                  # 15+15+5
    assert index.chunks[0].line_indices == list(range(15))
    assert index.chunks[-1].embedding is None      # 임베딩 없음


def test_search_ranks_by_keyword_overlap():
    doc = _doc([
        "오늘은 자바 상속과 다형성을 배웁니다",   # chunk 0 (보통)
        "전혀 무관한 잡담 내용",                  # 같은 chunk
    ] + ["채우기"] * 20 + [
        "상속 다형성 추상클래스 인터페이스 상속",  # 뒤쪽 chunk: 쿼리와 강한 매칭
    ])
    index = embedder.build_index(doc, chunk_lines=15)
    top = embedder.search(index, "상속 다형성", top_k=1)
    assert top and "상속" in top[0].text


def test_search_fallback_when_no_match():
    doc = _doc(["가나다", "라마바"])
    index = embedder.build_index(doc, chunk_lines=15)
    res = embedder.search(index, "전혀없는키워드xyz", top_k=2)
    assert res == index.chunks[:2]                 # 매칭 전무 → 앞 청크 폴백

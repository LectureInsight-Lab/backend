"""tests/test_behavior_tagger.py — keyword-BoW 태깅 (Paper #3)."""
from __future__ import annotations

from app.analysis import behavior_tagger as bt
from app.analysis.schemas import LectureDocument, Utterance


def _doc(texts: list[str]) -> LectureDocument:
    lines = [
        Utterance(timestamp="09:00:00", speaker_id="a", text=t, seconds_from_start=i * 5)
        for i, t in enumerate(texts)
    ]
    return LectureDocument(lecture_date="2026-02-02", instructor_id="i", all_lines=lines)


def test_normalize_min_evidence_returns_zero():
    # 근거(positive+negative)가 min_evidence(2) 미만 → 0
    assert bt.normalize_bow_score(1, 0, scale=5.0, min_evidence=2) == 0.0


def test_normalize_monotonic():
    low = bt.normalize_bow_score(0, 4)      # 부정 우세 → 낮음
    high = bt.normalize_bow_score(4, 0)     # 긍정 우세 → 높음
    assert 1.0 <= low < high <= 5.0


def test_positive_keyword_count():
    doc = _doc(["오늘은 목표를 안내합니다", "이번 시간 배울 내용은", "관련 없는 문장"])
    prof = bt.tag(doc)
    # 항목 4(학습 목표 안내) 긍정 지표가 잡힘
    assert prof.items[4].positive_count >= 2
    assert prof.items[4].bow_score > 1.0


def test_consecutive_repeat_counts_as_negative_item1():
    doc = _doc(["그 그 그 다음에", "정상 문장 입니다"])
    prof = bt.tag(doc)
    # 동일 어절 3연속 → 항목1 negative 가중
    assert prof.items[1].negative_count >= 1


def test_missing_indicator_item_scores_zero():
    doc = _doc(["아무 키워드 없는 평범한 문장", "또 다른 문장"])
    prof = bt.tag(doc)
    # 지표 사전에 없는 항목(예: 6)은 근거 0 → bow_score 0
    assert prof.items[6].bow_score == 0.0

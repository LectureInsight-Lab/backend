"""tests/test_ensemble.py — discrete vs high_inference 앙상블 (Paper #4)."""
from __future__ import annotations

from app.analysis import ensemble
from app.analysis.schemas import BehaviorProfile, ItemBoW, LLMItemRaw
from app.core.checklist import load_checklist

CL = load_checklist()


def _raw(item_id: int, score: float, conf: float) -> LLMItemRaw:
    return LLMItemRaw(item_id=item_id, score=score, evidence="e", strengths="s",
                      improvements="i", confidence=conf)


def _profile(item_id: int, bow_score: float) -> BehaviorProfile:
    return BehaviorProfile(lecture_date="d", instructor_id="i",
                           items={item_id: ItemBoW(bow_score=bow_score)})


def test_discrete_uses_llm_and_boosts_confidence():
    # 항목 4 = discrete
    raw = _raw(4, 4.0, 0.7)
    out = ensemble.ensemble_item(raw, _profile(4, 0.0), CL, confidence_threshold=0.5)
    assert out.final_score == 4.0
    assert out.final_confidence == 0.8            # 0.7 + 0.10


def test_high_inference_blends_when_bow_present():
    # 항목 1 = high_inference, bow_score=5 → 0.7*3 + 0.3*5 = 3.6
    raw = _raw(1, 3.0, 0.8)
    out = ensemble.ensemble_item(raw, _profile(1, 5.0), CL)
    assert abs(out.final_score - 3.6) < 1e-6
    assert out.final_confidence == 0.8            # 근거 있으면 신뢰도 유지


def test_high_inference_penalizes_confidence_when_no_bow():
    raw = _raw(1, 3.0, 0.8)
    out = ensemble.ensemble_item(raw, _profile(1, 0.0), CL)
    assert out.final_score == 3.0                 # LLM 단독
    assert abs(out.final_confidence - 0.68) < 1e-6  # 0.8 * 0.85


def test_needs_human_review_flag():
    raw = _raw(1, 3.0, 0.4)
    out = ensemble.ensemble_item(raw, _profile(1, 0.0), CL, confidence_threshold=0.5)
    assert out.needs_human_review is True


def test_ensemble_all_sorted_18():
    raws = [_raw(it.id, 3.0, 0.6) for it in CL.items]
    prof = BehaviorProfile(lecture_date="d", instructor_id="i", items={})
    scores = ensemble.ensemble_all(raws, prof, CL)
    assert len(scores) == 18
    assert [s.item_id for s in scores] == sorted(s.item_id for s in scores)

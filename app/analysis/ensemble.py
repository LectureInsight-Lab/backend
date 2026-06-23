"""앙상블 스코어링 (4단계) — Paper #4.

LLM 원본 점수와 BoW 점수를 항목 유형(discrete vs high_inference)에 따라 결합.

규칙::

    discrete 항목 (LLM 강세):
        final_score = llm_score
        final_confidence = min(llm_confidence + 0.10, 1.0)

    high_inference 항목 (BoW 보조):
        if bow_score > 0:
            final_score = 0.70 * llm_score + 0.30 * bow_score
        else:
            final_score = llm_score
            final_confidence = llm_confidence * 0.85   # 근거 부족 → 신뢰도 하향

리포트 표시:
    final_confidence < CONFIDENCE_THRESHOLD (기본 0.5) → needs_human_review=True
"""
from __future__ import annotations

from app.analysis.schemas import BehaviorProfile, ItemBoW, ItemScore, LLMItemRaw
from app.core.checklist import Checklist, ItemType
from app.core.config import settings


def ensemble_item(
    raw: LLMItemRaw,
    behavior: BehaviorProfile,
    checklist: Checklist,
    llm_weight: float = settings.ensemble_llm_weight,
    bow_weight: float = settings.ensemble_bow_weight,
    confidence_threshold: float = settings.confidence_threshold,
) -> ItemScore:
    """단일 항목 앙상블."""
    item = checklist.by_id(raw.item_id)
    bow: ItemBoW = behavior.items.get(raw.item_id, ItemBoW())
    bow_score = bow.bow_score

    if item.item_type == ItemType.DISCRETE:
        final_score = raw.score
        final_confidence = min(raw.confidence + 0.10, 1.0)
    else:  # HIGH_INFERENCE
        if bow_score > 0:
            final_score = llm_weight * raw.score + bow_weight * bow_score
            final_confidence = raw.confidence
        else:
            final_score = raw.score
            final_confidence = raw.confidence * 0.85

    final_score = round(max(1.0, min(5.0, final_score)), 4)
    final_confidence = round(max(0.0, min(1.0, final_confidence)), 4)

    return ItemScore(
        item_id=item.id,
        name=item.name,
        category=item.category,
        item_type=item.item_type.value,
        final_score=final_score,
        llm_score=raw.score,
        bow_score=bow_score,
        final_confidence=final_confidence,
        evidence=raw.evidence,
        strengths=raw.strengths,
        improvements=raw.improvements,
        needs_human_review=final_confidence < confidence_threshold,
    )


def ensemble_all(
    raws: list[LLMItemRaw],
    behavior: BehaviorProfile,
    checklist: Checklist,
    **kwargs,
) -> list[ItemScore]:
    """18개 항목 일괄 앙상블 (항목 id 오름차순 정렬)."""
    scores = [ensemble_item(raw, behavior, checklist, **kwargs) for raw in raws]
    return sorted(scores, key=lambda s: s.item_id)

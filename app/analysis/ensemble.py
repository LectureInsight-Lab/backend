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
    confidence < CONFIDENCE_THRESHOLD (기본 0.5) → "⚠ 인간 검토 권장"
"""
from app.analysis.schemas import BehaviorProfile, ItemScore, LLMItemRaw
from app.core.checklist import Checklist, ItemType


def ensemble_item(
    raw: LLMItemRaw,
    behavior: BehaviorProfile,
    checklist: Checklist,
    llm_weight: float = 0.70,
    bow_weight: float = 0.30,
    confidence_threshold: float = 0.5,
) -> ItemScore:
    """단일 항목 앙상블 (placeholder)."""
    raise NotImplementedError


def ensemble_all(
    raws: list[LLMItemRaw],
    behavior: BehaviorProfile,
    checklist: Checklist,
    **kwargs,
) -> list[ItemScore]:
    """18개 항목 일괄 앙상블 (placeholder)."""
    raise NotImplementedError

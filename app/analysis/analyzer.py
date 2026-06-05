"""LLM 분석 (3단계) — Paper #2 기반.

18개 항목에 대해 비동기 병렬로 LLM을 호출하고 구조화 JSON 응답을 받는다.

각 호출은:
    1) ChecklistItem 의 context_strategy 에 따라 컨텍스트를 결정
       - intro / middle / outro → 해당 구간 발화
       - full_sample / keyword  → embedder.search(쿼리, top_k)
    2) BehaviorProfile 에서 해당 항목 BoW 카운트 추출
    3) templates.build_messages 로 프롬프트 조립
    4) OpenAI Chat Completion (response_format=json_object) 호출
    5) LLMItemRaw 로 파싱

TODO:
- asyncio.gather 로 18개 병렬화
- tenacity 재시도 (rate limit / transient error)
- LLM 응답 캐싱 (.cache/llm)
- temperature=0.2 고정 (Paper #6)
"""
from app.analysis.schemas import (
    BehaviorProfile,
    LectureDocument,
    LectureIndex,
    LLMItemRaw,
)
from app.core.checklist import Checklist, ChecklistItem


async def analyze_item(
    item: ChecklistItem,
    document: LectureDocument,
    index: LectureIndex,
    behavior: BehaviorProfile,
) -> LLMItemRaw:
    """단일 항목 LLM 분석 (placeholder)."""
    raise NotImplementedError


async def analyze_lecture(
    document: LectureDocument,
    index: LectureIndex,
    behavior: BehaviorProfile,
    checklist: Checklist,
) -> list[LLMItemRaw]:
    """18개 항목 병렬 분석 (placeholder).

    구현 예::
        return await asyncio.gather(
            *[analyze_item(it, document, index, behavior) for it in checklist.items]
        )
    """
    raise NotImplementedError

"""LangChain 기반 분석 체인.

각 분석 항목별 프롬프트를 로드해 LCEL 체인으로 구성하고,
구조화 출력(Pydantic) 으로 결과를 받는다.

TODO:
- LLM 인스턴스 팩토리 (provider 분기: openai / anthropic)
- 프롬프트 로더 (configs/prompts/*.yaml)
- 청크 분할 입력 → 맵리듀스 결합
- LLM 캐시 연결
"""
from app.analysis.schemas import CriterionResult


def run_criterion(criterion: str, transcript: str) -> CriterionResult:
    """단일 분석 항목에 대해 LLM 체인을 실행 (placeholder)."""
    raise NotImplementedError

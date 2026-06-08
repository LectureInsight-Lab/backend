"""보조 통계 (BoW 입력 지표).

filler_word_ratio    → 항목 1 (불필요한 반복 표현)
avg_line_gap_seconds → 항목 12 (발화 속도 적절성)

LLM 호출 없이 빠르게 산출 가능한 객관 지표만 다룬다.
"""
from app.analysis.schemas import Utterance


FILLER_WORDS = ["어", "음", "그", "자", "이제", "뭐"]


def filler_word_ratio(utterances: list[Utterance]) -> float:
    """전체 단어 중 필러 단어 비율 (placeholder)."""
    raise NotImplementedError


def avg_line_gap_seconds(utterances: list[Utterance]) -> float:
    """발화 행 간 평균 시간 간격 (placeholder)."""
    raise NotImplementedError


def basic_stats(utterances: list[Utterance]) -> dict:
    """발화량/어휘 다양성 등 요약 통계 (placeholder)."""
    raise NotImplementedError

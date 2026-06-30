"""보조 통계 (BoW 입력 지표).

filler_word_ratio    → 항목 1 (불필요한 반복 표현)
avg_line_gap_seconds → 항목 12 (발화 속도 적절성)

LLM 호출 없이 빠르게 산출 가능한 객관 지표만 다룬다.
"""
from __future__ import annotations

import re
from collections import Counter

from app.analysis.schemas import Utterance


FILLER_WORDS: tuple[str, ...] = ("어", "음", "그", "자", "이제", "뭐", "아", "예")

# 어절(공백 토큰) 양 끝 구두점만 제거하기 위한 패턴
_PUNCT_RE = re.compile(r"[\.\,\?\!\~。，？！\"'\(\)\[\]\{\}…]+")

# 갭 임계값: 세션 내 짧은 침묵까지만 평균에 포함. 점심/세션간 갭은 제외.
DEFAULT_GAP_CAP_SECONDS = 30 * 60   # 30분


def _tokenize(text: str) -> list[str]:
    """공백 분리 후 양끝 구두점만 제거."""
    tokens = []
    for raw in text.split():
        stripped = _PUNCT_RE.sub("", raw)
        if stripped:
            tokens.append(stripped)
    return tokens


def filler_word_ratio(utterances: list[Utterance]) -> float:
    """전체 어절 중 필러 단어 비율 (0.0~1.0).

    한국어 필러는 단독 어절로 나오는 경우가 대부분이라
    "단어 == 필러" 매칭만 한다 (부분 일치 시 '그래서' 같은 어절을 오탐).
    """
    if not utterances:
        return 0.0
    total_tokens = 0
    filler_tokens = 0
    fillers = set(FILLER_WORDS)
    for u in utterances:
        for tok in _tokenize(u.text):
            total_tokens += 1
            if tok in fillers:
                filler_tokens += 1
    return filler_tokens / total_tokens if total_tokens else 0.0


def avg_line_gap_seconds(
    utterances: list[Utterance],
    gap_cap_seconds: int = DEFAULT_GAP_CAP_SECONDS,
) -> float:
    """인접 발화 간 평균 시간 간격(초).

    ``gap_cap_seconds`` 이상의 갭(세션 경계, 점심 휴식 등)은 평균에서 제외.
    """
    if len(utterances) < 2:
        return 0.0
    gaps: list[int] = []
    for prev, cur in zip(utterances, utterances[1:]):
        gap = cur.seconds_from_start - prev.seconds_from_start
        if 0 <= gap < gap_cap_seconds:
            gaps.append(gap)
    return sum(gaps) / len(gaps) if gaps else 0.0


def basic_stats(utterances: list[Utterance]) -> dict:
    """발화량/어휘 다양성 등 요약 통계.

    반환 키:
        line_count          전체 발화 줄 수
        token_count         전체 어절 수
        unique_token_count  서로 다른 어절 수
        type_token_ratio    unique / total (어휘 다양성)
        top_speaker         가장 많이 등장한 speaker_id
        speaker_count       서로 다른 speaker_id 수
    """
    if not utterances:
        return {
            "line_count": 0,
            "token_count": 0,
            "unique_token_count": 0,
            "type_token_ratio": 0.0,
            "top_speaker": None,
            "speaker_count": 0,
        }
    token_counter: Counter[str] = Counter()
    speaker_counter: Counter[str] = Counter()
    for u in utterances:
        token_counter.update(_tokenize(u.text))
        speaker_counter[u.speaker_id] += 1
    total_tokens = sum(token_counter.values())
    unique_tokens = len(token_counter)
    return {
        "line_count": len(utterances),
        "token_count": total_tokens,
        "unique_token_count": unique_tokens,
        "type_token_ratio": (unique_tokens / total_tokens) if total_tokens else 0.0,
        "top_speaker": speaker_counter.most_common(1)[0][0],
        "speaker_count": len(speaker_counter),
    }

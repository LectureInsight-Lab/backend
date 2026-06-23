"""BoW 행동 태깅 (2-A단계) — Paper #3.

모든 발화 행을 순회하며 18개 항목별 긍정/부정 지표 출현을 카운트한다.
``configs/bow_indicators.yaml`` 의 키워드 사전을 사용.

LLM이 데이터가 적을 때 흔들리는 신뢰도를 BoW가 보조한다(ensemble.py 에서 가중 혼합).

설계:
- 키워드 매칭은 라인 단위 1회 카운트(같은 라인에서 한 키워드가 여러 번 나와도 1회).
  → STT 반복 발화로 인한 과대집계 방지. "지표 출현 횟수" 의미와 일치.
- 항목 1(불필요한 반복)은 yaml negative(필러) 외에 "동일 단어 3회 이상 연속" 규칙을
  추가 negative 로 가중 (yaml 주석의 별도 규칙).
- raw_score = positive_count - negative_count
  bow_score = 1 + 4 * sigmoid(raw_score / scale)   (1.0 ~ 5.0)
  단, positive+negative < min_evidence 이면 bow_score=0.0 (근거 부족 → LLM 점수만 사용).
- yaml 에 지표가 없는 항목(3·6·7·9·10·11·13·14·15 등)은 자동으로 카운트 0 → bow_score 0.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import yaml

from app.analysis.schemas import BehaviorProfile, ItemBoW, LectureDocument

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOW_CONFIG = PROJECT_ROOT / "configs" / "bow_indicators.yaml"

ITEM_IDS = range(1, 19)
_DEFAULT_SCALE = 5.0
_DEFAULT_MIN_EVIDENCE = 2
_CONSECUTIVE_REPEAT = 3   # 항목 1: 동일 단어 N회 이상 연속 → 반복 negative


@lru_cache(maxsize=1)
def load_indicators(path: str | Path = BOW_CONFIG) -> dict:
    """``configs/bow_indicators.yaml`` 로드 (캐싱)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _count_keywords(lines: list[str], keywords: list[str]) -> int:
    """키워드가 등장한 라인 수 합 (라인당 키워드별 1회)."""
    if not keywords:
        return 0
    total = 0
    for text in lines:
        for kw in keywords:
            if kw and kw in text:
                total += 1
    return total


def _consecutive_repeat_count(lines: list[str], run_length: int = _CONSECUTIVE_REPEAT) -> int:
    """동일 어절이 ``run_length`` 회 이상 연속된 구간 수 (항목 1 가중)."""
    hits = 0
    for text in lines:
        tokens = text.split()
        run = 1
        for prev, cur in zip(tokens, tokens[1:]):
            if cur == prev:
                run += 1
                if run == run_length:    # 런이 길이에 도달한 순간 1회 카운트
                    hits += 1
            else:
                run = 1
    return hits


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def normalize_bow_score(
    positive: int,
    negative: int,
    scale: float = _DEFAULT_SCALE,
    min_evidence: int = _DEFAULT_MIN_EVIDENCE,
) -> float:
    """raw count → 1~5 정규화 점수.

    근거(positive+negative)가 ``min_evidence`` 미만이면 0.0 반환
    (ensemble 에서 LLM 점수만 사용하도록).
    """
    if positive + negative < min_evidence:
        return 0.0
    raw = positive - negative
    return round(1.0 + 4.0 * sigmoid(raw / scale), 4)


def tag(document: LectureDocument) -> BehaviorProfile:
    """LectureDocument → 18 항목 BoW 프로필."""
    cfg = load_indicators()
    indicators: dict = cfg.get("indicators", {})
    norm = cfg.get("normalization", {})
    scale = float(norm.get("scale", _DEFAULT_SCALE))
    min_evidence = int(norm.get("min_evidence", _DEFAULT_MIN_EVIDENCE))

    lines = [u.text for u in document.all_lines]

    items: dict[int, ItemBoW] = {}
    for item_id in ITEM_IDS:
        spec = indicators.get(item_id) or indicators.get(str(item_id)) or {}
        positive = _count_keywords(lines, spec.get("positive") or [])
        negative = _count_keywords(lines, spec.get("negative") or [])

        # 항목 1: 동일 단어 연속 반복을 부정 신호로 추가 가중
        if item_id == 1:
            negative += _consecutive_repeat_count(lines)

        items[item_id] = ItemBoW(
            positive_count=positive,
            negative_count=negative,
            bow_score=normalize_bow_score(positive, negative, scale, min_evidence),
        )

    return BehaviorProfile(
        lecture_date=document.lecture_date,
        instructor_id=document.instructor_id,
        items=items,
    )

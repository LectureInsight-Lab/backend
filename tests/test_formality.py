"""
tests/test_formality.py — 항목 3 말투 분류(formality) 회귀 테스트

[이수민 - 2026-06-15]
검증:
  - classify_formality: 종결 표면형 → 격식/비격식/반말/중립
      · 습니다/합니다 → 격식존댓말
      · 어요/죠/거든요 → 비격식존댓말
      · 야/어/잖아 → 반말
      · 된다/한다(한다체) → 중립, EC·None → 중립
  - formality_profile: consistency_ratio / violation_count / dominant

문자열 기반이라 Mecab/kss 불필요(스키마만 사용).
"""
from __future__ import annotations

from app.analysis.schemas import Sentence
from app.preprocessing.item03_consistency import (
    BANMAL, FORMAL, INFORMAL_POLITE, NEUTRAL,
    classify_formality, formality_profile,
)


def _sent(morph: str | None, tag: str | None) -> Sentence:
    return Sentence(
        text="x", segment_index=0, start_timestamp="09:00:00", end_timestamp="09:00:00",
        start_seconds=0, end_seconds=0, ending_morph=morph, ending_tag=tag,
    )


def test_classify_formal():
    assert classify_formality("습니다", "VV+EF") == FORMAL
    assert classify_formality("합니다", "XSV+EF") == FORMAL
    assert classify_formality("습니까", "EF") == FORMAL


def test_classify_informal_polite():
    assert classify_formality("어요", "EF") == INFORMAL_POLITE
    assert classify_formality("죠", "EF") == INFORMAL_POLITE
    assert classify_formality("거든요", "EF") == INFORMAL_POLITE
    assert classify_formality("잖아요", "EF") == INFORMAL_POLITE


def test_classify_banmal():
    assert classify_formality("야", "VCP+EF") == BANMAL
    assert classify_formality("어", "EF") == BANMAL
    assert classify_formality("잖아", "EF") == BANMAL


def test_classify_neutral_declarative_and_nonef():
    assert classify_formality("된다", "VV+EF") == NEUTRAL     # 한다체
    assert classify_formality("한다", "VV+EF") == NEUTRAL
    assert classify_formality("뭐냐면", "VCP+EC") == NEUTRAL   # EC(불완결)
    assert classify_formality(None, None) == NEUTRAL


def test_profile_dominant_jondaetmal():
    sents = [_sent("습니다", "EF"), _sent("어요", "EF"),
             _sent("야", "VCP+EF"), _sent("된다", "VV+EF")]
    prof = formality_profile(sents)
    assert prof["jondaetmal"] == 2     # 격식1 + 비격식1
    assert prof["banmal"] == 1
    assert prof["neutral"] == 1        # 된다 한다체 → 중립(분모 제외)
    assert round(prof["consistency_ratio"], 1) == 66.7   # 2/3
    assert prof["violation_count"] == 1
    assert prof["dominant"] == "jondaetmal"


def test_profile_empty():
    prof = formality_profile([])
    assert prof["consistency_ratio"] == 0.0
    assert prof["dominant"] is None

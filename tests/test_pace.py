"""
tests/test_pace.py — 항목 12 발화 속도(pace) 회귀 테스트 — 음절(SPM) 헤드라인

[이수민 - 2026-06-15 / 팀 안건] 헤드라인 단위 어절 → 음절(SPM) 전환 반영.
검증:
  - syllable_count(헤드라인) / eojeol_count(보조)
  - lecture_minutes: 큰 갭(점심) 제외
  - spm_kr / spm_speaking: 음절 / 시간
  - window_spm: 5분 구간 음절/분, 희박 구간 제외
  - score_band: 아나운서 355 기준 1~5 (최적 215~300, slowdown 시 5)
  - pace_profile: 헤드라인/보조 키

순수 산술이라 Mecab/kss 불필요.
"""
from __future__ import annotations

import pytest

from app.analysis.schemas import Utterance
from app.preprocessing import pace


def _u(sec: int, text: str) -> Utterance:
    return Utterance(timestamp="00:00:00", speaker_id="a", text=text, seconds_from_start=sec)


def test_counts():
    assert pace.syllable_count("오늘은 자바를 배웁니다") == 10   # 음절
    assert pace.eojeol_count("오늘은 자바를 배웁니다") == 3      # 어절(보조)
    assert pace.syllable_count("JPA 21 입니다") == 3            # 영어/숫자 누락 → 입,니,다


def test_lecture_minutes_excludes_big_gap():
    utts = [_u(0, "가 나"), _u(60, "다 라"), _u(2060, "마 바")]  # 2000초 갭 제외 → 1분
    assert pace.lecture_minutes(utts) == pytest.approx(1.0)


def test_spm_kr_basic():
    # span 120초(=2분), 음절 4 → 2 spm
    utts = [_u(0, "가나"), _u(120, "다라")]
    assert pace.spm_kr(utts) == pytest.approx(2.0)


def test_window_spm_filters_sparse():
    utts = [_u(10, "가" * 150)]                 # 150음절 → 150/5 = 30
    assert pace.window_spm(utts, min_syllable=150) == [30.0]
    assert pace.window_spm([_u(10, "가 나")], min_syllable=150) == []  # 2음절 제외


def test_score_band_optimal_and_slowdown():
    assert pace.score_band(255) == 4                      # 최적이나 slowdown 미입력 → 4
    assert pace.score_band(255, slowdown=0.80) == 5       # 핵심 감속 → 5
    assert pace.score_band(255, slowdown=0.95) == 4       # 감속 없음 → 4


def test_score_band_off_optimal_symmetry():
    assert pace.score_band(205) == 4 and pace.score_band(320) == 4   # 195~215 / 300~340
    assert pace.score_band(180) == 3 and pace.score_band(360) == 3   # 160~195 / 340~390
    assert pace.score_band(140) == 2 and pace.score_band(420) == 2   # 125~160 / 390~445
    assert pace.score_band(100) == 1 and pace.score_band(500) == 1   # <125 / >445


def test_pace_profile_keys_and_diagnostic():
    utts = [_u(0, "하나 둘 셋"), _u(60, "넷 다섯")]
    prof = pace.pace_profile(utts)
    assert {"spm_speaking", "spm_kr", "score", "window_std",
            "wpm_kr", "syllable_per_eojeol", "pct_of_announcer"} <= set(prof)
    assert prof["syllable_count"] == 7    # 하·나·둘·셋·넷·다·섯 = 7 음절
    assert prof["eojeol_count"] == 5


def test_slowdown_none_without_ranges():
    utts = [_u(0, "가 나"), _u(60, "다 라")]
    assert pace.slowdown_ratio(utts, key_ranges=[]) is None

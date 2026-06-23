"""
tests/test_comprehension_check.py — 항목 16 이해 확인 질문 회귀 테스트

[이수민 - 2026-06-17]
검증:
  - detect_checks: 실데이터 마커('되셨어요' 등) 의문형만 탐지, 설명체 '이해'는 미탐지
  - check_rate: check_count / 강의 시간(분), 큰 갭 제외
  - timing_ratio: 앵커 직후 5분 이내 비율, 앵커 없으면 None
  - score_band: check_rate 1~5 + timing_bonus(±1), clip [1,5]

순수 regex·산술이라 Mecab/kss 불필요.
"""
from __future__ import annotations

from app.analysis.schemas import Utterance
from app.preprocessing import comprehension_check as cc


def _u(sec: int, text: str) -> Utterance:
    return Utterance(timestamp="00:00:00", speaker_id="a", text=text, seconds_from_start=sec)


# ── 탐지 정밀도 ───────────────────────────────────────────────
def test_detects_real_marker():
    assert cc.check_count([_u(0, "자 여기까지 되셨어요")]) == 1
    assert cc.check_count([_u(0, "이해되셨나요")]) == 1
    assert cc.check_count([_u(0, "다들 알겠어요")]) == 1
    assert cc.check_count([_u(0, "이거 맞죠")]) == 1
    assert cc.check_count([_u(0, "화면 보이시죠")]) == 1


def test_ignores_explanatory_and_declarative():
    # 설명체 '이해' — 질문 아님
    assert cc.check_count([_u(0, "이걸 이해를 하고 다음으로 넘어가요")]) == 0
    # 평서 안심 '어렵지 않아요' — 의문형 아님
    assert cc.check_count([_u(0, "이건 그렇게 어렵지 않아요")]) == 0
    # 평서 '됩니다' — 확인 아님
    assert cc.check_count([_u(0, "이렇게 하면 됩니다")]) == 0


def test_line_level_single_count():
    # 한 라인에 마커가 둘이어도 라인 단위 1회
    assert cc.check_count([_u(0, "이해되셨어요 다들 알겠어요")]) == 1


def test_pattern_breakdown():
    prof = cc.comprehension_profile([_u(0, "되셨어요"), _u(30, "맞죠"), _u(60, "되셨어요")])
    assert prof["check_count"] == 3
    assert prof["pattern_breakdown"]["되셨/됐"] == 2
    assert prof["pattern_breakdown"]["맞죠/맞나요"] == 1


# ── check_rate ────────────────────────────────────────────────
def test_check_rate_basic():
    # span 600초(=10분), 확인 1회 → 0.1/분
    utts = [_u(0, "되셨어요"), _u(600, "자 다음")]
    assert cc.check_rate(utts) == 0.1


def test_check_rate_excludes_big_gap():
    # 점심 갭(2000초)은 강의 시간에서 제외 → 1분, 확인 1회 → 1.0/분
    utts = [_u(0, "되셨어요"), _u(60, "다음"), _u(2060, "오후 시작")]
    assert cc.check_rate(utts) == 1.0


# ── timing_ratio ──────────────────────────────────────────────
def test_timing_ratio_none_without_anchor():
    hits = cc.detect_checks([_u(100, "되셨어요")])
    assert cc.timing_ratio(hits, anchor_seconds=[]) is None


def test_timing_ratio_window():
    hits = cc.detect_checks([_u(100, "되셨어요"), _u(1000, "알겠어요")])
    # 앵커 0초: 100초 확인은 OK(5분 이내), 1000초 확인은 NG → 50%
    assert cc.timing_ratio(hits, anchor_seconds=[0]) == 50.0


def test_proxy_anchor_detection():
    utts = [_u(0, "예를 들어서 설명할게요"), _u(60, "직접 해보세요")]
    assert len(cc.detect_anchors(utts)) == 2


# ── 채점 ──────────────────────────────────────────────────────
def test_check_rate_score_bands():
    assert cc.check_rate_score(0.12) == 5
    assert cc.check_rate_score(0.08) == 4
    assert cc.check_rate_score(0.06) == 3
    assert cc.check_rate_score(0.04) == 2
    assert cc.check_rate_score(0.01) == 1


def test_timing_bonus_and_final_clip():
    assert cc.timing_bonus(80) == 1
    assert cc.timing_bonus(20) == -1
    assert cc.timing_bonus(50) == 0
    assert cc.timing_bonus(None) == 0
    # 5점 + 보너스 → 5로 클립
    assert cc.final_score(0.12, 80) == 5
    # 1점 - 보너스 → 1로 클립
    assert cc.final_score(0.01, 20) == 1


# ── 진짜 대기 (습관 vs 진짜 확인) ─────────────────────────────
def test_detect_checks_records_gap_and_endline():
    utts = [_u(0, "오늘은 자바를 배웁니다"), _u(10, "되셨어요"), _u(40, "다음으로")]
    hits = cc.detect_checks(utts)
    assert len(hits) == 1
    assert hits[0].gap_after == 30          # 40 - 10
    assert hits[0].ends_line is True        # '되셨어요'가 라인 끝


def test_ends_line_false_when_check_midline():
    # 확인 표현 뒤에 설명이 더 붙으면 라인 끝 아님
    hits = cc.detect_checks([_u(0, "되셨어요 그러면 이제 다음 내용으로 넘어가서 설명할게요")])
    assert hits[0].ends_line is False


def test_wait_threshold_uses_floor():
    # 촘촘한 강의(갭 1~2초)는 floor(8초)로 보호
    utts = [_u(i * 2, "가 나 다") for i in range(20)]
    assert cc.wait_threshold(utts) == 8.0


def test_wait_threshold_uses_percentile_when_above_floor():
    # 갭이 10초 균일 → p90 = 10 > floor 8 → 10
    utts = [_u(i * 10, "가 나 다") for i in range(20)]
    assert cc.wait_threshold(utts) == 10.0


def test_genuine_wait_filters_habit():
    # 임계 10초: 끊고 30초 멈춘 확인 = 진짜 / 2초 만에 이어간 확인 = 습관
    genuine = _u(100, "되셨어요")           # gap 30 → 진짜
    habit = _u(130, "되셨어요")             # gap 2 → 습관
    utts = [genuine, habit, _u(132, "바로 이어서 설명하면 이렇게 되고요")]
    # baseline 리듬(앞에 10초 갭들) 깔아 임계 ~10초 만들기
    pad = [_u(i * 10, "가 나 다 라 마") for i in range(10)]
    utts = pad + [_u(200, "되셨어요"), _u(230, "다음"), _u(231, "되셨어요"),
                  _u(233, "바로 이어 설명하면 이렇게 되는 거예요")]
    g, thr = cc.genuine_checks(utts)
    assert thr >= 8.0
    # 230-200=30 → 진짜 1건, 233-231=2 → 제외
    assert len(g) == 1 and g[0].seconds_from_start == 200


def test_profile_exposes_genuine_fields():
    pad = [_u(i * 10, "가 나 다 라 마") for i in range(10)]
    utts = pad + [_u(200, "되셨어요"), _u(230, "다음 내용")]
    prof = cc.comprehension_profile(utts)
    assert {"genuine_count", "genuine_ratio", "genuine_check_rate",
            "genuine_score", "wait_threshold_seconds"} <= set(prof)
    assert prof["genuine_count"] == 1

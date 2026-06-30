"""
preprocessing/item12_pace.py — 항목 12 발화 속도 적절성: SPM(음절/분) 헤드라인 + 어절 보조

[이수민 - 2026-06-15] 최초 — 어절 기준 WPM_kr.

[이수민 - 2026-06-15 / 팀 안건 통과] 헤드라인 단위를 어절 → 음절(SPM)로 전환.
  어절 함수(wpm_kr 등)는 보조(진단)로 강등. 이유:
   - 외부 앵커: 아나운서 발화 SPM ≈ 355 (남 356.9±6.5 / 여 353.1±4.5) — 단일 강사 편향을
     빼는 모집단 기준. 어절엔 한국어 출판 기준이 없음.
     출처: https://jslhd.org/_common/do.php?a=full&b=12&bidx=2656&aidx=30072
   - STT 강건성: 어절=띄어쓰기 토큰이라 STT 띄어쓰기 오류에 흔들림. 음절=한글 글자 수라 무관.
   - 문헌 호환: 한국어 발화속도 연구는 SPM/음절·초 기준.
  실측(15강의): 음절/어절 ≈ 2.7, 발화 SPM ≈ 255 = 아나운서의 72% (매우 일관).
  주의: 음절은 영어/숫자 용어(JPA, API 등)를 누락 → 코딩 강의서 약간 과소집계 가능.

지표:
  spm_kr        = 음절 / 강의 시간(분)        # 강의 시간 = 세션 span (점심 등 큰 갭 제외)
  spm_speaking  = 음절 / 말하는 시간(분)      # 헤드라인 (아나운서 SPM과 동일 축: 발화 시간)
  window_spm    = 5분 구간별 음절/분          # 변동성
  slowdown      = 핵심 구간 SPM / 전체 평균    # Step 3 (핵심 구간 감속, <0.85=감속)
  score_band    = 아나운서 355 기준 1~5 채점

채점(score_band, spm_speaking 기준):
  5: 215~300 + 핵심 감속(slowdown<0.85) / 4: 215~300(균일)·195~215·300~340
  3: 160~195·340~390 / 2: 125~160·390~445 / 1: <125·>445
  ⚠️ 최적 창(215~300=아나운서의 60~85%)의 *위치*는 제 교육학 판단 — 강의-이해도 연구/
     전문가 라벨로 확정 예정. 스케일/천장(아나운서 355)은 출판 근거 기반.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter

from app.analysis.schemas import Utterance

# 아나운서 발화 SPM 기준 (남 356.9±6.5 / 여 353.1±4.5). 모집단 앵커.
ANNOUNCER_SPM = 355
# 한글 음절 블록 (가–힣)
_SYLLABLE_RE = re.compile(r"[가-힣]")

# 강의 시간에서 제외할 '큰 갭'(점심/세션 휴식) 기준
BREAK_GAP_SECONDS = 30 * 60
# 말하는 시간 근사용: 이보다 큰 갭은 침묵으로 보고 제외
SPEAKING_GAP_CAP_SECONDS = 20
WINDOW_MINUTES = 5
# 5분 구간 중 발화 희박 구간(경계/휴식) 제외용 최소 음절
_MIN_WINDOW_SYLLABLE = 150


# ── 카운트 ────────────────────────────────────────────────────
def syllable_count(text: str) -> int:
    """음절(한글 글자) 수 — 헤드라인 단위."""
    return len(_SYLLABLE_RE.findall(text))


def eojeol_count(text: str) -> int:
    """어절(띄어쓰기 토큰) 수 — 보조/진단용."""
    return len(text.split())


# ── 시간 (단위 무관) ──────────────────────────────────────────
def lecture_minutes(utterances: list[Utterance], break_gap_seconds: int = BREAK_GAP_SECONDS) -> float:
    """강의 시간(분) = 전체 span - 큰 갭(점심/휴식) 합."""
    if len(utterances) < 2:
        return 0.0
    secs = [u.seconds_from_start for u in utterances]
    span = secs[-1] - secs[0]
    breaks = sum(b - a for a, b in zip(secs, secs[1:]) if (b - a) > break_gap_seconds)
    return max(span - breaks, 0) / 60


def speaking_minutes(utterances: list[Utterance], cap_seconds: int = SPEAKING_GAP_CAP_SECONDS) -> float:
    """말하는 시간 근사(분) = 갭 <= cap 인 구간만 합 (긴 침묵 제외)."""
    secs = [u.seconds_from_start for u in utterances]
    return sum(g for a, b in zip(secs, secs[1:]) if 0 <= (g := b - a) <= cap_seconds) / 60


# ── 헤드라인: 음절 SPM ────────────────────────────────────────
def spm_kr(utterances: list[Utterance], break_gap_seconds: int = BREAK_GAP_SECONDS) -> float:
    """음절 / 강의 시간(분)."""
    minutes = lecture_minutes(utterances, break_gap_seconds)
    if minutes <= 0:
        return 0.0
    return sum(syllable_count(u.text) for u in utterances) / minutes


def spm_speaking(utterances: list[Utterance], cap_seconds: int = SPEAKING_GAP_CAP_SECONDS) -> float:
    """음절 / 말하는 시간(분). 헤드라인 — 아나운서 SPM과 동일 축(발화 시간)."""
    minutes = speaking_minutes(utterances, cap_seconds)
    if minutes <= 0:
        return 0.0
    return sum(syllable_count(u.text) for u in utterances) / minutes


def window_spm(
    utterances: list[Utterance],
    window_min: int = WINDOW_MINUTES,
    min_syllable: int = _MIN_WINDOW_SYLLABLE,
) -> list[float]:
    """5분(기본) 구간별 음절/분 리스트. 발화 희박 구간(경계/휴식) 제외."""
    width = window_min * 60
    bins: Counter[int] = Counter()
    for u in utterances:
        bins[u.seconds_from_start // width] += syllable_count(u.text)
    return [c / window_min for c in bins.values() if c >= min_syllable]


# ── 보조: 어절 WPM (진단·교차검증용) ──────────────────────────
def wpm_kr(utterances: list[Utterance], break_gap_seconds: int = BREAK_GAP_SECONDS) -> float:
    """[보조] 어절 / 강의 시간(분). 헤드라인은 spm_kr."""
    minutes = lecture_minutes(utterances, break_gap_seconds)
    if minutes <= 0:
        return 0.0
    return sum(eojeol_count(u.text) for u in utterances) / minutes


# ── slowdown (비율이라 단위 무관 — 음절 기준) ─────────────────
def slowdown_ratio(
    utterances: list[Utterance],
    key_ranges: list[tuple[int, int]],
    break_gap_seconds: int = BREAK_GAP_SECONDS,
) -> float | None:
    """핵심 구간 SPM / 전체 평균 SPM. < 0.85 면 감속(15%+).

    ``key_ranges`` = 핵심 개념 구간들의 (start_sec, end_sec). 항목 2-4/3-1 산출물 필요.
    범위가 없으면 None.
    """
    if not key_ranges:
        return None
    overall = spm_kr(utterances, break_gap_seconds)
    if overall <= 0:
        return None
    key_syl = key_seconds = 0
    for u, nxt in zip(utterances, utterances[1:]):
        if any(lo <= u.seconds_from_start < hi for lo, hi in key_ranges):
            key_syl += syllable_count(u.text)
            dur = nxt.seconds_from_start - u.seconds_from_start
            if 0 <= dur <= SPEAKING_GAP_CAP_SECONDS:
                key_seconds += dur
    if key_seconds == 0:
        return None
    return (key_syl / (key_seconds / 60)) / overall


# ── 채점 (아나운서 355 기준) ──────────────────────────────────
def score_band(spm: float, slowdown: float | None = None) -> int:
    """발화 SPM → 1~5 점. 아나운서 355 기준 밴드.

    최적(215~300) 안에서 핵심 구간 감속(slowdown<0.85)이 확인되면 5, 아니면 4(보수적).
    slowdown 미입력(현재)이면 최적 강의는 4점. 밴드 위치는 ⚠️ 확정 전(docstring 참고).
    """
    if spm < 125 or spm > 445:
        return 1
    if spm < 160 or spm > 390:
        return 2
    if spm < 195 or spm > 340:
        return 3
    if spm < 215 or spm > 300:
        return 4
    # 215~300 최적
    return 5 if (slowdown is not None and slowdown < 0.85) else 4


# ── 종합 ──────────────────────────────────────────────────────
def run(df) -> dict:
    """kss DataFrame → 발화 속도 점수 (항목 12). 파이프라인 레지스트리 진입점."""
    from app.preprocessing.utils import build_utterances

    p = pace_profile(build_utterances(df))
    reason = (
        f"발화 속도 분당 {p['spm_speaking']:.0f}음절"
        f" (아나운서 대비 {p['pct_of_announcer']:.0f}%)"
    )
    return {"final_score": p["score"], "reason": reason, "evidence": None}


def pace_profile(utterances: list[Utterance]) -> dict:
    """발화 속도 종합. 점수는 score_band 로 산출(원지표 + 점수 함께 노출).

    헤드라인(음절):
        spm_kr / spm_speaking / window_spm(+mean/median/std) / score
    보조(어절): wpm_kr / 진단(syllable·eojeol·ratio)
    """
    windows = window_spm(utterances)
    w_mean = statistics.fmean(windows) if windows else 0.0
    w_median = statistics.median(windows) if windows else 0.0
    w_std = statistics.pstdev(windows) if len(windows) > 1 else 0.0

    syl = sum(syllable_count(u.text) for u in utterances)
    eoj = sum(eojeol_count(u.text) for u in utterances)
    spm_spk = spm_speaking(utterances)
    return {
        # 헤드라인 (음절)
        "spm_kr": spm_kr(utterances),
        "spm_speaking": spm_spk,
        "window_spm": windows,
        "window_mean": w_mean,
        "window_median": w_median,
        "window_std": w_std,
        "score": score_band(spm_spk),            # slowdown 미연결 → 최적은 4
        "pct_of_announcer": (spm_spk / ANNOUNCER_SPM * 100) if spm_spk else 0.0,
        # 보조 (어절) + 진단
        "wpm_kr": wpm_kr(utterances),
        "syllable_count": syl,
        "eojeol_count": eoj,
        "syllable_per_eojeol": (syl / eoj) if eoj else 0.0,
        "lecture_minutes": lecture_minutes(utterances),
    }

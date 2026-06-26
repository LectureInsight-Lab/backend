"""
항목 1-1: 불필요한 반복 표현 (repetition)

입력: raw STT .txt 파일 경로
  형식: <HH:MM:SS> speaker_id: text

설계 절차 (강의_품질_평가루브릭_설계.md § 1-1):
  1. 전체 어절 수 계산 (text.split() — 공통 정의: 어절 = 띄어쓰기 단위)
  2. 불필요 단어 사전 매칭 → filler_count
       [간투사]   음, 어, 아, 에, 저(Kiwi IC 태그), 뭐(Kiwi IC 태그)
       [담화표지] 이제, 근데, 그니까, 어떻게 보면, 이런 식으로
       [접속남용] 그래서·그리고 (창 내 3회 초과 시만 계산)
  3. FWR(%) = filler_count / total_words × 100
  4. 분당 발생 수 = filler_count / 강의 시간(분)
  5. 슬라이딩 윈도우: 크기=5분, 스텝=2.5분 (50% 오버랩) → 창별 FWR
  6. window_max_fwr = 창별 FWR 최댓값

채점 (결정 테이블):
  base_score: 전체 FWR 기준
    5 ≤ 3% / 4 ≤ 6% / 3 ≤ 10% / 2 ≤ 15% / 1 > 15%
  window_max_fwr 패널티:
    5점 → > 8%이면 4점 / 4점 → > 10%이면 3점 / 3점 → > 15%이면 2점

출력:
  {"evidence": list|null, "reason": str, "final_score": int|"N/A"}
  mode=""      → evidence: null
  mode="debug" → evidence: 상세 지표
"""
from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

ITEM_ID     = 1
ITEM_NAME   = "불필요한 반복 표현"
CATEGORY_ID = 1

# ── 불필요 단어 사전 (설계 문서 § 1-1) ─────────────────────────────────────
# [간투사] 및 [담화표지] — 토큰 단위 정확 매칭 (항상 계산)
_FILLER_TOKENS: frozenset[str] = frozenset({
    "음", "어", "아", "에",          # [간투사]
    "이제", "근데", "그니까",         # [담화표지]
})

# [간투사] Kiwi IC(감탄사) 태그로만 필러 인정 (품사 문맥 의존)
_FILLER_IC_FORMS: frozenset[str] = frozenset({"저", "뭐"})

# [담화표지] 다중 어절 — 원문 서브스트링 매칭
_FILLER_PHRASES: tuple[str, ...] = ("어떻게 보면", "이런 식으로")

# [접속남용] — 창 내 3회 초과 시 전체 횟수를 필러로 계산
_CONNECTIVE_ABUSE: frozenset[str] = frozenset({"그래서", "그리고"})
_CONNECTIVE_THRESHOLD = 3

# 슬라이딩 윈도우 파라미터
_WINDOW_SEC = 5 * 60   # 5분
_STEP_SEC   = 150      # 2.5분 (50% 오버랩)

_LINE_RE = re.compile(r"^<(\d{2}:\d{2}:\d{2})>\s+\S+:\s*(.*)$")


# ── STT 파싱 ─────────────────────────────────────────────────────────────────

def _ts_to_sec(ts: str) -> int:
    h, m, s = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s


def _parse_lines(txt_path: Path) -> list[tuple[int, str]]:
    """(running_sec, text) 리스트. 12시간 시계 delta 보정 포함."""
    raw = txt_path.read_text(encoding="utf-8").splitlines()
    parsed: list[tuple[int, str]] = []
    running_sec = 0
    prev_abs: int | None = None

    for line in raw:
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        text = m.group(2).strip()
        if not text:
            continue
        abs_sec = _ts_to_sec(m.group(1))
        if prev_abs is None:
            prev_abs = abs_sec
        delta = abs_sec - prev_abs
        if delta < -3600:      # 12시간 시계 넘김 보정
            delta += 12 * 3600
        elif delta < 0:
            delta = 0
        running_sec += delta
        prev_abs = abs_sec
        parsed.append((running_sec, text))

    return parsed


# ── 발화별 필러 통계 사전 계산 ───────────────────────────────────────────────

def _build_utterance_stats(parsed: list[tuple[int, str]]) -> list[dict]:
    """
    Kiwi를 한 번만 초기화하여 발화별 통계를 미리 계산.
    각 항목: {sec, words, basic, connective}
      basic       = IC필러 + 토큰필러 + 구필러
      connective  = 그래서·그리고 총 횟수 (창 레벨에서 임계값 적용)
    """
    try:
        from kiwipiepy import Kiwi
        kiwi: object | None = Kiwi()
    except Exception:
        logger.warning("[item01] Kiwi 로드 실패 — IC 판별을 토큰 매칭으로 대체")
        kiwi = None

    stats: list[dict] = []

    for sec, text in parsed:
        tokens = text.split()
        words  = len(tokens)

        # [담화표지] 다중 어절
        phrase_count = sum(text.count(ph) for ph in _FILLER_PHRASES)

        # [간투사·담화표지] 단순 토큰 매칭
        token_filler = sum(1 for t in tokens if t in _FILLER_TOKENS)

        # [간투사] Kiwi IC 태그 (저, 뭐)
        if kiwi is not None:
            ic_count = sum(
                1 for tok in kiwi.tokenize(text)
                if tok.tag == "IC" and tok.form in _FILLER_IC_FORMS
            )
        else:
            ic_count = sum(1 for t in tokens if t in _FILLER_IC_FORMS)

        # [접속남용]
        connective = sum(1 for t in tokens if t in _CONNECTIVE_ABUSE)

        stats.append({
            "sec":        sec,
            "words":      words,
            "basic":      token_filler + ic_count + phrase_count,
            "connective": connective,
        })

    return stats


# ── 전체 FWR 계산 ─────────────────────────────────────────────────────────────

def _global_fwr(stats: list[dict]) -> tuple[int, int, float, float]:
    """
    (total_words, filler_count, fwr_pct, filler_per_min) 반환.
    접속어는 전체 횟수를 포함 (전체 구간에서의 누적 남용 반영).
    """
    total_words  = sum(s["words"]      for s in stats)
    total_basic  = sum(s["basic"]      for s in stats)
    total_conn   = sum(s["connective"] for s in stats)
    filler_count = total_basic + total_conn

    total_sec = stats[-1]["sec"] if stats else 0
    total_min = max(total_sec / 60, 1.0)

    fwr = filler_count / total_words * 100 if total_words else 0.0
    fpm = filler_count / total_min

    return total_words, filler_count, round(fwr, 2), round(fpm, 2)


# ── 슬라이딩 윈도우 최대 FWR ─────────────────────────────────────────────────

def _window_max_fwr(stats: list[dict]) -> float:
    """5분 창, 2.5분 스텝 슬라이딩 윈도우 최대 FWR(%)."""
    if not stats:
        return 0.0

    total_duration = stats[-1]["sec"]
    max_fwr = 0.0
    win_start = 0

    while win_start <= total_duration:
        win_end = win_start + _WINDOW_SEC
        window  = [s for s in stats if win_start <= s["sec"] < win_end]

        if window:
            words = sum(s["words"] for s in window)
            if words > 0:
                basic      = sum(s["basic"]      for s in window)
                connective = sum(s["connective"] for s in window)
                # 창 내 임계값 초과 시 전체 접속어 횟수를 필러로 간주
                abuse = connective if connective > _CONNECTIVE_THRESHOLD else 0
                fwr   = (basic + abuse) / words * 100
                max_fwr = max(max_fwr, fwr)

        win_start += _STEP_SEC

    return round(max_fwr, 2)


# ── 채점 ─────────────────────────────────────────────────────────────────────

def _base_score(fwr: float) -> int:
    """전체 FWR(%) → 기본 점수."""
    if fwr <= 3.0:
        return 5
    if fwr <= 6.0:
        return 4
    if fwr <= 10.0:
        return 3
    if fwr <= 15.0:
        return 2
    return 1


def _apply_penalty(score: int, win_max: float) -> int:
    """window_max_fwr 패널티 (결정 테이블)."""
    if score == 5 and win_max > 8.0:
        return 4
    if score == 4 and win_max > 10.0:
        return 3
    if score == 3 and win_max > 15.0:
        return 2
    return score


# ── 공개 API ─────────────────────────────────────────────────────────────────

def score_repetition(txt_path: str | Path, mode: str = "") -> dict:
    """
    Parameters
    ----------
    txt_path : str | Path
        raw STT .txt 파일 경로.
    mode : str
        "" → evidence: null / "debug" → evidence: 상세 지표
    """
    txt_path = Path(txt_path)
    parsed   = _parse_lines(txt_path)

    if not parsed:
        logger.warning(f"[item01] STT 파싱 결과 없음: {txt_path.name}")
        return _build(score="N/A", evidence_items=[], reason="STT 파싱 실패", mode=mode)

    logger.info(f"[item01] {txt_path.name}: {len(parsed)}개 발화")

    # 발화별 필러 통계 (Kiwi 1회 초기화)
    stats = _build_utterance_stats(parsed)

    # Step 1-4: 전체 FWR
    total_words, filler_count, fwr, fpm = _global_fwr(stats)

    # Step 5-6: 슬라이딩 윈도우 최대 FWR
    win_max = _window_max_fwr(stats)

    # 채점
    base  = _base_score(fwr)
    score = _apply_penalty(base, win_max)

    reason = (
        f"전체 FWR {fwr:.1f}%({base}점 기준)"
        f" / window_max_fwr {win_max:.1f}%"
        f" / 분당 {fpm:.1f}회"
        f" → 최종 {score}점"
    )
    logger.info(f"[item01] {reason}")

    evidence_items = [{
        "total_words":     total_words,
        "filler_count":    filler_count,
        "fwr_pct":         fwr,
        "filler_per_min":  fpm,
        "window_max_fwr":  win_max,
        "base_score":      base,
    }]

    return _build(score=score, evidence_items=evidence_items, reason=reason, mode=mode)


def _build(
    score: int | str,
    evidence_items: list[dict],
    reason: str = "",
    mode: str = "",
) -> dict:
    return {
        "evidence":    evidence_items if mode == "debug" else None,
        "reason":      reason,
        "final_score": score,
    }

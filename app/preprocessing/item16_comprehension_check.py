"""
preprocessing/item16_comprehension_check.py — 항목 16 이해 확인 질문: 탐지(regex) + check_rate/timing_ratio

[이수민 - 2026-06-17]
조작적 정의(루브릭 v2.0):
  강의 중 수강생의 이해 여부를 확인하는 질문의 분당 빈도(check_rate)와,
  개념 설명/예시/실습 직후 적절한 타이밍에 이루어지는 비율(timing_ratio).

  check_rate     = check_count / 강의 시간(분)        # 강의 시간 = pace.lecture_minutes (큰 갭 제외)
  timing_ratio   = (앵커 직후 5분 이내 확인 수) / check_count × 100
  score_band     = check_rate → 1~5 + timing_bonus(±1)

[데이터 발견 — 루브릭 예시 regex가 실제 강의와 안 맞음]
  루브릭 예시("이해하셨나요","질문 있으신가요","따라오시나요")는 15강의에서 거의 안 나옴.
  이 강사의 실제 이해 확인 마커는 **"되셨어요"(235회)** — "(이해/작업) 되셨어요?" 형태의 구어 확인.
  반대로 "이해"(413회)는 대부분 설명체("이해를 돕다")라 질문이 아님 → 단순 키워드 매칭은 과대.
  "어렵지 않~"도 대부분 평서 안심("어렵지 않아요")이라 *의문형*만 카운트.
  → 실데이터에서 관찰된 '학생을 향한 의문형 확인'만 고정밀 패턴으로 집계.

[STT 특성 주의]
  STT 라인엔 '?'가 거의 없고 발화가 라인 단위로 파편화됨. 따라서 문장부호가 아니라
  '확인 의문형 어미 패턴'으로 탐지한다. 한 라인에 패턴이 있으면 확인 1회로 센다(라인 단위).

[타이밍 앵커 의존성 — slowdown_ratio 와 동일 패턴]
  timing_ratio 의 정식 앵커(개념 정의 4-2 / 예시·실습 구간 종료)는 타 항목(9·13·14) 산출물.
  미연결 시 timing_ratio=None. EDA 에선 예시·실습 *cue 키워드*를 **프록시 앵커**로 써
  분포만 관찰한다(프록시임을 명시). 정식 앵커 확정 후 교체.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from app.analysis.schemas import Utterance
from app.preprocessing.item12_pace import lecture_minutes

# ── 이해 확인 질문 탐지 패턴 (실데이터 15강의 관찰 기반, 고정밀) ──────────
# 각 (이름, 정규식). 이름은 진단/분포 출력용. 모두 '학생을 향한 의문형 확인'.
_CHECK_SPECS: list[tuple[str, str]] = [
    # 핵심: 시-높임 확인(되셨/되시 + 종결). 학생 대상이라 신뢰. [개선 2026-06-28: 지/습니까/는지]
    ("되셨_높임",    r"(?:되셨|되시)(?:어요|나요|죠|지|어|습니까|을까요|는지)"),
    # bare 됐 — 의문형만(됐을까요/됐습니까/됐나요/됐냐). 평서 '됐어요'(상태서술)는 precision 위해 제외.
    # [골든셋 분석] 기존 '됐'+어요/어/죠가 bare 됐어요(상태서술) 115건을 헛집계 → 정밀도 누수 차단.
    ("됐_의문",      r"됐(?:을까요|습니까|나요|냐)"),
    # 이해 + 의문형: 이해되셨어요 / 이해하셨어요 / 이해 가셨어요 / 이해되시나요
    ("이해_의문",    r"이해\s*(?:가|는|를)?\s*(?:되셨|되시|하셨|가셨|되었|됐)\S*(?:어요|나요|죠|어)?"),
    # 이해 가다(구어): 이해 가죠/가지/가시죠/가셨/이해 갔어요/이해하죠 (이해 간다=평서는 제외) [개선]
    ("이해_가다",    r"이해\s*(?:가|갔|하)\s*(?:죠|지|시죠|시지|셨\S*|어요|나요|가요|가죠|갔\S*)"),
    # 이해 안 됨 확인: 이해 안 되시는 분 / 이해가 안 되시는 / 모르시는 분 / 헷갈리시는 분
    ("막힌사람",     r"(?:이해\s*(?:가)?\s*안\s*[되돼]|모르시는\s*분|안\s*되시는\s*분|헷갈리시는\s*분|막히시는\s*분)"),
    # 알겠어요/알겠죠/아시겠어요/아시겠죠/아시겠나요/알겠지 [개선: bare 지]
    ("알겠/아시겠",  r"(?:알겠|아시겠)(?:어요|나요|죠|어|지요|지)"),
    # 맞죠/맞나요/맞으시죠 (확인 동의 요청)
    ("맞죠/맞나요",  r"맞(?:죠|나요|으시죠|지요)"),
    # 괜찮으세요/괜찮으시죠/괜찮나요/괜찮죠
    ("괜찮으세요",   r"괜찮(?:으세요|으시죠|나요|죠|으신가요)"),
    # 보이시죠/보이시나요/보이세요? (화면 확인)
    ("보이시죠",     r"보이(?:시죠|시나요|세요|죠|나요)"),
    # 따라오시나요/따라오셨어요/따라오시죠
    ("따라오",       r"따라\s*오(?:시나요|셨어요|시죠|세요)"),
    # 질문 있/없 으신가요/으세요/어요
    ("질문있나요",   r"질문\s*(?:있|없)\S*(?:신가요|으세요|어요|나요|으신|으세요)"),
    # 어렵지 않으셨나요 / 어렵지 않으세요  ← 의문형만 (평서 '어렵지 않아요'는 제외)
    ("안어렵나요",   r"어렵지\s*않으(?:셨나요|세요|신가요|셨어요)"),
    # 여기까지 (오셨/하셨/해셨/만드셨/확인) — '여기까지' 앵커로 제한해 narration FP 방지 [개선]
    ("여기까지확인", r"여기까지\s*(?:오셨|하셨|해셨|만드셨|확인하셨|확인하세요|오세요|왔어요|이해)"),
]
_CHECK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pat)) for name, pat in _CHECK_SPECS
]

# ── 프록시 타이밍 앵커: 예시/실습 cue (EDA 전용, 정식 앵커는 타 항목 산출물) ──
_ANCHOR_SPECS: list[tuple[str, str]] = [
    ("예시", r"예(?:를\s*들|시\s*로|로\s*들|컨대)|비유\s*하?(?:면|자면)"),
    ("실습", r"실습|직접\s*해|한번\s*해보|해\s*볼게요|해보겠습니다|따라\s*(?:해|치)|타이핑|작성해\s*보"),
]
_ANCHOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pat)) for name, pat in _ANCHOR_SPECS
]

# 적절 타이밍: 앵커(개념/예시/실습) 종료 후 이 시간 이내 확인이면 OK
TIMING_WINDOW_SECONDS = 5 * 60

# ── 직후 갭 / '진짜 대기' 판정 (습관 vs 진짜 확인) ───────────────
# STT 타임스탬프는 발화 '시작점' → 인접 갭은 대부분 직전 발화 길이(전체 p50 ~10초)다.
# [데이터 발견 2026-06-17] 확인 직후 갭(p50 12s)은 baseline(p50 10s)보다 +2초뿐 →
# 대부분의 "되셨어요"는 별도 대기 없이 리듬에 묻힘(습관). 단 갭이 본인 리듬 상위권인
# 소수(전체 p90 19s, gap≥20s=확인의 13.5% vs baseline 8.4%)는 '진짜 대기' 후보.
# → '진짜 확인'은 강사 *본인 리듬* 대비 상위 갭(+ 라인 끝에서 끊음)으로 판정한다.
GAP_CAP_SECONDS = 120          # 세션/휴식 갭은 분석서 제외
WAIT_PERCENTILE = 90           # 강사 본인 라인 갭의 상위 10% = 대기 후보
WAIT_FLOOR_SECONDS = 8         # 절대 하한 (아주 촘촘한 강의 보호)
_TAIL_CHARS = 6                # 매칭이 라인 끝 이 글자 수 안이면 '라인 끝'으로 봄


@dataclass
class CheckHit:
    """탐지된 이해 확인 발화 1건."""

    seconds_from_start: int
    timestamp: str
    pattern: str          # 매칭된 패턴 이름 (진단용)
    text: str             # 원본 라인
    gap_after: int | None = None   # 다음 발화 시작까지 초 (직후 대기 근사)
    ends_line: bool = False        # 확인이 라인 끝 = 끊고 멈춤 후보


# ── 1) 탐지 ───────────────────────────────────────────────────
def _first_match(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> str | None:
    """패턴 목록 중 첫 매칭의 이름 반환(없으면 None). 라인 단위 1회 카운트."""
    for name, rx in patterns:
        if rx.search(text):
            return name
    return None


def _ends_line(text: str) -> bool:
    """확인 패턴 매칭이 라인 끝부분(_TAIL_CHARS 이내)에 있는가 = 확인하고 끊었나."""
    last_end = max(
        (m.end() for _, rx in _CHECK_PATTERNS for m in rx.finditer(text)),
        default=0,
    )
    return last_end >= len(text.rstrip()) - _TAIL_CHARS


def detect_checks(utterances: list[Utterance]) -> list[CheckHit]:
    """이해 확인 발화 탐지. 한 라인에 패턴이 있으면 확인 1회(라인 단위).

    각 발화에 직후 갭(gap_after, 다음 발화 시작까지 초)·라인 끝 여부(ends_line)를
    함께 기록 — '습관 vs 진짜 대기' 판정 입력.
    """
    hits: list[CheckHit] = []
    n = len(utterances)
    for i, u in enumerate(utterances):
        name = _first_match(u.text, _CHECK_PATTERNS)
        if not name:
            continue
        nxt = utterances[i + 1] if i + 1 < n else None
        gap_after = (nxt.seconds_from_start - u.seconds_from_start) if nxt else None
        hits.append(
            CheckHit(
                seconds_from_start=u.seconds_from_start,
                timestamp=u.timestamp,
                pattern=name,
                text=u.text,
                gap_after=gap_after,
                ends_line=_ends_line(u.text),
            )
        )
    return hits


# ── '진짜 대기' 확인 판정 (강사 본인 리듬 기준) ───────────────
def line_gaps(utterances: list[Utterance], cap: int = GAP_CAP_SECONDS) -> list[int]:
    """인접 발화 시작 간 갭(초). 세션/휴식 등 cap 초과 갭은 제외 (baseline 리듬)."""
    secs = [u.seconds_from_start for u in utterances]
    return [g for a, b in zip(secs, secs[1:]) if 0 <= (g := b - a) <= cap]


def _percentile(values: list[float], q: float) -> float:
    """선형보간 percentile (numpy 없이)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (q / 100)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def wait_threshold(
    utterances: list[Utterance],
    percentile: int = WAIT_PERCENTILE,
    floor_seconds: int = WAIT_FLOOR_SECONDS,
) -> float:
    """진짜 대기 판정 기준 = max(강사 본인 갭 분포 상위 percentile, floor)."""
    return max(float(floor_seconds), _percentile(line_gaps(utterances), percentile))


def is_genuine_wait(hit: CheckHit, threshold: float) -> bool:
    """라인 끝에서 확인하고(끊고) 본인 리듬 상위 갭만큼 멈췄으면 '진짜 대기'."""
    return hit.ends_line and hit.gap_after is not None and hit.gap_after >= threshold


def genuine_checks(
    utterances: list[Utterance],
    percentile: int = WAIT_PERCENTILE,
    floor_seconds: int = WAIT_FLOOR_SECONDS,
) -> tuple[list[CheckHit], float]:
    """'진짜 대기' 확인만 추림 + 사용된 임계값 반환."""
    thr = wait_threshold(utterances, percentile, floor_seconds)
    return [h for h in detect_checks(utterances) if is_genuine_wait(h, thr)], thr


def detect_anchors(utterances: list[Utterance]) -> list[int]:
    """[EDA 프록시] 예시/실습 cue 발화의 경과초 리스트 (정식 앵커 대용)."""
    return [
        u.seconds_from_start
        for u in utterances
        if _first_match(u.text, _ANCHOR_PATTERNS)
    ]


# ── 2) 지표 ───────────────────────────────────────────────────
def check_count(utterances: list[Utterance]) -> int:
    return len(detect_checks(utterances))


def check_rate(utterances: list[Utterance]) -> float:
    """check_count / 강의 시간(분). 강의 시간은 pace.lecture_minutes(큰 갭 제외)."""
    minutes = lecture_minutes(utterances)
    if minutes <= 0:
        return 0.0
    return check_count(utterances) / minutes


def timing_ratio(
    check_hits: list[CheckHit],
    anchor_seconds: list[int],
    window_seconds: int = TIMING_WINDOW_SECONDS,
) -> float | None:
    """앵커(개념/예시/실습) 직후 window 이내 확인 비율(%).

    anchor_seconds 가 비면 None (정식 앵커 미연결 — slowdown_ratio 와 동일 규약).
    """
    if not check_hits or not anchor_seconds:
        return None
    anchors = sorted(anchor_seconds)
    ok = 0
    for h in check_hits:
        # h 직전(<=)의 가장 가까운 앵커를 찾아 window 안이면 OK
        prev = [a for a in anchors if a <= h.seconds_from_start]
        if prev and (h.seconds_from_start - prev[-1]) <= window_seconds:
            ok += 1
    return ok / len(check_hits) * 100


# ── 3) 채점 (루브릭 v2.0) ─────────────────────────────────────
def check_rate_score(rate: float) -> int:
    """check_rate → 1~5 (Rosenshine 2012: 10~15분당 1회 확인)."""
    if rate >= 0.10:
        return 5
    if rate >= 0.07:
        return 4
    if rate >= 0.05:
        return 3
    if rate >= 0.03:
        return 2
    return 1


def timing_bonus(ratio: float | None) -> int:
    """timing_ratio ≥70% → +1, <30% → -1, 그 외/미연결 → 0."""
    if ratio is None:
        return 0
    if ratio >= 70:
        return 1
    if ratio < 30:
        return -1
    return 0


def final_score(rate: float, ratio: float | None) -> int:
    """final = clip(check_rate_score + timing_bonus, 1, 5)."""
    raw = check_rate_score(rate) + timing_bonus(ratio)
    return max(1, min(5, raw))


# ── 4) 종합 ───────────────────────────────────────────────────
def run(df) -> dict:
    """kss DataFrame → 이해 확인 질문 점수 (항목 16). 파이프라인 레지스트리 진입점."""
    from app.preprocessing.utils import build_utterances

    p = comprehension_profile(build_utterances(df), use_proxy_anchors=True)
    reason = (
        f"이해 확인 표현 {p['check_count']}회"
        f" (분당 {p['check_rate']:.2f}회)"
    )
    return {"final_score": p["final_score"], "reason": reason, "evidence": None}


def comprehension_profile(
    utterances: list[Utterance],
    anchor_seconds: list[int] | None = None,
    use_proxy_anchors: bool = False,
) -> dict:
    """항목 16 종합. 원지표 + 점수 함께 노출(점수 밴드는 잠정).

    anchor_seconds 미입력 시: use_proxy_anchors=True 면 예시/실습 cue 프록시 앵커 사용,
    아니면 timing_ratio=None.
    """
    hits = detect_checks(utterances)
    if anchor_seconds is None:
        anchor_seconds = detect_anchors(utterances) if use_proxy_anchors else []
    minutes = lecture_minutes(utterances)
    rate = (len(hits) / minutes) if minutes > 0 else 0.0
    t_ratio = timing_ratio(hits, anchor_seconds)

    # 습관 vs 진짜 대기 — 강사 본인 리듬 상위 갭 + 라인 끝
    thr = wait_threshold(utterances)
    genuine = [h for h in hits if is_genuine_wait(h, thr)]
    genuine_rate = (len(genuine) / minutes) if minutes > 0 else 0.0

    pattern_breakdown: dict[str, int] = {}
    for h in hits:
        pattern_breakdown[h.pattern] = pattern_breakdown.get(h.pattern, 0) + 1

    return {
        # ── raw (탐지된 모든 확인 표현) ──
        "check_count": len(hits),
        "lecture_minutes": minutes,
        "check_rate": rate,
        "timing_ratio": t_ratio,
        "check_rate_score": check_rate_score(rate),
        "timing_bonus": timing_bonus(t_ratio),
        "final_score": final_score(rate, t_ratio),
        # ── 진짜 대기 보정 (습관 추임새 제외) ──
        "wait_threshold_seconds": thr,
        "genuine_count": len(genuine),
        "genuine_ratio": (len(genuine) / len(hits) * 100) if hits else 0.0,
        "genuine_check_rate": genuine_rate,
        "genuine_score": check_rate_score(genuine_rate),
        # ── 진단 ──
        "pattern_breakdown": pattern_breakdown,
        "anchor_count": len(anchor_seconds),
        "hits": hits,
    }

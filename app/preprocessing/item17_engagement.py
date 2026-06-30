"""
preprocessing/item17_engagement.py — 항목 17 참여 유도: 유도 발화 탐지 + 침묵 갭 + 결과 확인 피드백

[이수민 - 2026-06-17]
조작적 정의(루브릭 v2.0):
  수강생이 직접 시도하도록 유도하는 발화 횟수(engagement_count), 유도 후 실제 시도
  시간(침묵 갭, avg_gap), 시도 후 결과 확인 발화 유무(feedback).

  engagement_count = '직접 해보라'는 학생 명령형 발화 수
  gap_after        = 유도 발화 → 다음 강사 발화 시작까지 초 (학생 시도/대기 근사)
  feedback         = 유도 후 5분 이내 '결과 확인' 발화 유무
  final_score      = min(5, round(engagement_score + gap_bonus + feedback_bonus))

[설계 — regex(후보) + 타임스탬프(진짜 판단), item 16과 동일 철학]
  regex 는 '유도 표현 후보'를 찾는 거친 1차 필터일 뿐. 진짜 신호는 행동(침묵 갭)이다.
  STT 타임스탬프는 발화 '시작점'이라 갭은 침묵이 아니라 직전 발화 길이(전체 p50 ~10초).
  → '진짜 실습 넘김'은 유도 후 갭이 *실습 최소 시도 시간(30초, 루브릭 근거)* 이상인 경우.

[데이터 발견 — 15강의]
  ① spec 예시 regex 는 좁아 과소(0.3배): "해 보세요"·"한번 해봐요"·"써 보세요" 변형 누락.
     또 "직접"을 단독으로 쓰면 "직접 호출"(기술 용어)을 오탐 → 학생 명령형(~보세요/봐요)만.
  ② 강사 본인 시범(해볼게요/해보도록 하겠습니다/해봅시다)은 '세요/봐요' 어미로 자동 제외.
  ③ 유도 후 갭: 68%가 <15초, ≥30초(진짜 시도)는 ~6%뿐 → "해보세요" 하고 안 기다림(말뿐).
  ④ 결과 확인 피드백: 루브릭 예시("어떻게 됐어요/완성하셨")는 실데이터 ~0건. 단순 "되셨어요/
     됐어요"는 item 16의 *말버릇* tic 이라 결과 확인으로 세면 48%로 부풀려짐 → tic 제외 시 ~4%.
     따라서 본 모듈 feedback 은 '결과 확인' 의미가 분명한 표현만 카운트(tic 제외).

근거: Flanders(1960) 교사 독점 발화 > 70% = 저품질 → 유도 최소 2회를 3점 하한.
      30초 갭 = 실습 최소 시도 시간(타이핑 50WPM, 20~30자 코드 ≈ 25~35초).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import fmean, median

from app.analysis.schemas import Utterance
from app.preprocessing.item12_pace import lecture_minutes

# ── 참여 유도 탐지: '학생이 지금 직접 시도'하도록 시키는 명령형 ─────────
# 강사 본인 시범(보자/봅시다/볼게요/보도록/보겠/보면/보시면)은 세요/봐요 어미라 자동 제외.
# [개선 2026-06-28] 골든셋 분석으로 recall 갭 보강: _ACTION 확장 + STT 잘린 어미(보세)·bare 봐·어라 허용.
_ACTION = (
    r"(?:해|풀어|작성해|구현해|만들어|짜|쳐|입력해|실행해|돌려|적어|써|출력해|"
    r"호출해|테스트해|선언해|완성해|구해|정해|넣어|생각해|바꿔|고쳐|확인해|로드해|"
    r"체크해|추출해|사용해|인서트해|선택해|복사해|눌러|지워|열어|클릭해|가져와)"
)
_ENGAGE_SPECS: list[tuple[str, str]] = [
    # action + 보세요/십시오/실래요/보세(STT 잘림) — 해보세요/짜보세요/해보세
    ("직접시도",   _ACTION + r"\s*보(?:세요|십시오|실래요|세)(?![가-힣])"),
    # action + 봐요/봐(요 잘림) — 해봐요/써봐/넣어봐/생각해봐. (봐야/봐서 연결어미는 제외)
    ("직접시도2",  _ACTION + r"\s*봐(?:요)?(?![가-힣])"),
    # action + 어라 명령형 — 넣어라/풀어라/봐라
    ("명령_어라",  r"(?:넣|풀|써|적|만들|짜|쳐|돌려|해)어?라(?![가-힣])"),
    # 혼자/스스로 + 시도
    ("혼자시도",   r"(?:혼자|스스로)\s*\S{0,10}?\s*보(?:세요|십시오|세)(?![가-힣])|"
                  r"(?:혼자|스스로)\s*\S{0,10}?\s*봐(?:요)?(?![가-힣])"),
]
_ENGAGE_PATTERNS = [(n, re.compile(p)) for n, p in _ENGAGE_SPECS]

# ── 결과 확인 피드백: '시도 결과를 확인'하는 의미가 분명한 표현만 ──────
# (item 16의 단순 "되셨어요/됐어요" tic 은 제외 — 결과 확인이 아니라 습관 추임새)
_FEEDBACK_SPECS: list[tuple[str, str]] = [
    ("결과확인", r"어떻게\s*됐|결과가\s*(?:나왔|어때|어떻게|어떠)|나왔(?:어요|나요|죠)"),
    ("완성확인", r"완성(?:하셨|되셨|됐)|다\s*(?:하셨|짜셨|작성하셨|만드셨|푸셨)"),
    ("에러확인", r"에러\s*(?:없|안|나)|오류\s*(?:없|안|나)"),
    ("성공확인", r"잘\s*(?:되셨|됐|나왔|푸셨|짜셨)"),
]
_FEEDBACK_PATTERNS = [(n, re.compile(p)) for n, p in _FEEDBACK_SPECS]

# ── 임계값 ────────────────────────────────────────────────────
GAP_CAP_SECONDS = 30 * 60       # 세션 분리(점심/실습블록) 이상 갭은 분석서 제외
GENUINE_GAP_SECONDS = 30        # 진짜 시도 넘김 = 유도 후 이 시간 이상 침묵 (루브릭 근거)
FEEDBACK_WINDOW_SECONDS = 5 * 60
GAP_BONUS_SECONDS = 15          # 루브릭 gap_bonus 기준


@dataclass
class EngagementHit:
    """탐지된 참여 유도 발화 1건."""

    seconds_from_start: int
    timestamp: str
    pattern: str
    text: str
    gap_after: int | None = None    # 다음 강사 발화까지 초 (학생 시도/대기 근사)
    has_feedback: bool = False      # 유도 후 5분 이내 결과 확인 발화 유무
    feedback_text: str | None = None


# ── 탐지 ──────────────────────────────────────────────────────
def _first_match(text: str, patterns) -> str | None:
    for name, rx in patterns:
        if rx.search(text):
            return name
    return None


def detect_engagements(utterances: list[Utterance]) -> list[EngagementHit]:
    """참여 유도 발화 탐지 + 직후 갭·5분 내 결과 확인 피드백 부착 (라인 단위 1회)."""
    hits: list[EngagementHit] = []
    n = len(utterances)
    for i, u in enumerate(utterances):
        name = _first_match(u.text, _ENGAGE_PATTERNS)
        if not name:
            continue
        nxt = utterances[i + 1] if i + 1 < n else None
        gap_after = (nxt.seconds_from_start - u.seconds_from_start) if nxt else None

        t0 = u.seconds_from_start
        fb_text = None
        for v in utterances[i + 1:]:
            if v.seconds_from_start <= t0:
                continue
            if v.seconds_from_start > t0 + FEEDBACK_WINDOW_SECONDS:
                break
            if _first_match(v.text, _FEEDBACK_PATTERNS):
                fb_text = v.text
                break

        hits.append(
            EngagementHit(
                seconds_from_start=u.seconds_from_start,
                timestamp=u.timestamp,
                pattern=name,
                text=u.text,
                gap_after=gap_after,
                has_feedback=fb_text is not None,
                feedback_text=fb_text,
            )
        )
    return hits


# ── 갭 (행동 신호) ────────────────────────────────────────────
def line_gaps(utterances: list[Utterance], cap: int = GAP_CAP_SECONDS) -> list[int]:
    """인접 발화 시작 간 갭(초), 세션 분리 cap 초과 제외 (baseline 리듬)."""
    secs = [u.seconds_from_start for u in utterances]
    return [g for a, b in zip(secs, secs[1:]) if 0 <= (g := b - a) <= cap]


def avg_gap(hits: list[EngagementHit], cap: int = GAP_CAP_SECONDS) -> float:
    """유도 후 평균 침묵 갭(초). 세션 분리 이상 갭은 제외."""
    gaps = [h.gap_after for h in hits if h.gap_after is not None and h.gap_after <= cap]
    return fmean(gaps) if gaps else 0.0


def genuine_handoff_count(hits: list[EngagementHit], threshold: int = GENUINE_GAP_SECONDS,
                          cap: int = GAP_CAP_SECONDS) -> int:
    """진짜 시도 넘김 수 = 유도 후 갭이 threshold~cap 인 경우 (실제 침묵을 준 유도)."""
    return sum(1 for h in hits
               if h.gap_after is not None and threshold <= h.gap_after <= cap)


# ── 채점 (루브릭 v2.0) ─────────────────────────────────────────
def engagement_score(count: int) -> int:
    """유도 횟수 → 3/2/1 (Flanders: 최소 2회 = 3점 하한)."""
    if count >= 2:
        return 3
    if count == 1:
        return 2
    return 1


def gap_bonus(avg_gap_seconds: float) -> int:
    """avg_gap ≥ 15초 → +1 (측정 불가/미만 0)."""
    return 1 if avg_gap_seconds >= GAP_BONUS_SECONDS else 0


def feedback_bonus(feedback_count: int) -> int:
    """결과 확인 피드백 있음 → +1."""
    return 1 if feedback_count > 0 else 0


def final_score(count: int, avg_gap_seconds: float, feedback_count: int) -> int:
    """final = min(5, round(engagement_score + gap_bonus + feedback_bonus))."""
    raw = engagement_score(count) + gap_bonus(avg_gap_seconds) + feedback_bonus(feedback_count)
    return max(1, min(5, round(raw)))


# ── 종합 ──────────────────────────────────────────────────────
def run(df) -> dict:
    """kss DataFrame → 참여 유도 점수 (항목 17). 파이프라인 레지스트리 진입점."""
    from app.preprocessing.utils import build_utterances

    p = engagement_profile(build_utterances(df))
    reason = (
        f"참여 유도 발화 {p['engagement_count']}회"
        f" (평균 대기 {p['avg_gap']:.0f}초, 피드백 {p['feedback_count']}회)"
    )
    return {"final_score": p["final_score"], "reason": reason, "evidence": None}


def engagement_profile(utterances: list[Utterance]) -> dict:
    """항목 17 종합. 원지표 + 점수 + 행동 신호(진짜 넘김) 함께 노출."""
    hits = detect_engagements(utterances)
    minutes = lecture_minutes(utterances)
    a_gap = avg_gap(hits)
    fb_count = sum(1 for h in hits if h.has_feedback)
    genuine = genuine_handoff_count(hits)
    base_gaps = line_gaps(utterances)

    pattern_breakdown: dict[str, int] = {}
    for h in hits:
        pattern_breakdown[h.pattern] = pattern_breakdown.get(h.pattern, 0) + 1

    return {
        # ── 원지표 ──
        "engagement_count": len(hits),
        "lecture_minutes": minutes,
        "avg_gap": a_gap,
        "feedback_count": fb_count,
        "feedback_rate": (fb_count / len(hits) * 100) if hits else 0.0,
        # ── 채점 (루브릭) ──
        "engagement_score": engagement_score(len(hits)),
        "gap_bonus": gap_bonus(a_gap),
        "feedback_bonus": feedback_bonus(fb_count),
        "final_score": final_score(len(hits), a_gap, fb_count),
        # ── 행동 신호 (진짜 시도 넘김) ──
        "genuine_handoff_count": genuine,
        "genuine_ratio": (genuine / len(hits) * 100) if hits else 0.0,
        "gap_ge15": sum(1 for h in hits if h.gap_after is not None and h.gap_after >= 15),
        "baseline_gap_median": median(base_gaps) if base_gaps else 0.0,
        # ── 진단 ──
        "pattern_breakdown": pattern_breakdown,
        "hits": hits,
    }

"""
tests/test_engagement.py — 항목 17 참여 유도 회귀 테스트

[이수민 - 2026-06-17]
검증:
  - detect_engagements: 학생 명령형(~보세요/봐요)만, 강사 본인 시범(해볼게요 등) 제외
  - 직접/한번 해 등 오탐 방지 (직접 호출 = 기술용어, 해볼게요 = 강사)
  - gap_after / avg_gap: 유도 후 침묵 갭, 세션 분리 갭 제외
  - genuine_handoff: 30초 이상 침묵만 '진짜 시도 넘김'
  - feedback: 결과 확인 표현만(되셨어요 tic 제외), 5분 윈도우
  - score_band: engagement/gap/feedback 보너스

순수 regex·산술이라 Mecab/kss 불필요.
"""
from __future__ import annotations

from app.analysis.schemas import Utterance
from app.preprocessing import engagement as eng


def _u(sec: int, text: str) -> Utterance:
    return Utterance(timestamp="00:00:00", speaker_id="a", text=text, seconds_from_start=sec)


# ── 탐지 정밀도 ───────────────────────────────────────────────
def test_detects_student_imperative():
    assert eng.detect_engagements([_u(0, "이거 한번 해보세요")])
    assert eng.detect_engagements([_u(0, "직접 코드를 작성해보세요")])
    assert eng.detect_engagements([_u(0, "혼자 한번 짜봐요")])


def test_excludes_instructor_self_and_terms():
    # 강사 본인 시범
    assert not eng.detect_engagements([_u(0, "제가 한번 해볼게요")])
    assert not eng.detect_engagements([_u(0, "이거 한번 해보도록 하겠습니다")])
    assert not eng.detect_engagements([_u(0, "같이 해봅시다")])
    # 기술 용어 (직접 호출) — 유도 아님
    assert not eng.detect_engagements([_u(0, "이건 직접 호출할 수 없어요")])
    # 조건/설명 (해보시면)
    assert not eng.detect_engagements([_u(0, "이렇게 해보시면 결과가 나옵니다")])


# ── 갭 (행동 신호) ────────────────────────────────────────────
def test_gap_after_recorded():
    hits = eng.detect_engagements([_u(100, "짜보세요"), _u(140, "자 다음")])
    assert hits[0].gap_after == 40


def test_avg_gap_excludes_session_split():
    # 30분(1800초) 이상 갭은 세션 분리 → avg 에서 제외
    hits = eng.detect_engagements([_u(0, "해보세요"), _u(40, "음"), _u(60, "해보세요"), _u(3700, "오후 시작")])
    # 첫 유도 gap 40(포함), 둘째 유도 gap 3640(>cap 제외) → avg = 40
    assert eng.avg_gap(hits) == 40.0


def test_genuine_handoff_threshold():
    hits = eng.detect_engagements([_u(0, "해보세요"), _u(20, "다음"),      # gap 20 < 30 → 습관
                                   _u(100, "짜보세요"), _u(145, "확인")])  # gap 45 ≥ 30 → 진짜
    assert eng.genuine_handoff_count(hits) == 1


# ── 피드백 (결과 확인, tic 제외) ──────────────────────────────
def test_feedback_genuine_within_window():
    utts = [_u(0, "한번 풀어보세요"), _u(120, "자 다들 어떻게 됐어요")]
    hits = eng.detect_engagements(utts)
    assert hits[0].has_feedback is True


def test_feedback_excludes_habit_tic():
    # 단순 '되셨어요' tic 은 결과 확인으로 안 셈
    utts = [_u(0, "한번 해보세요"), _u(60, "자 되셨어요")]
    hits = eng.detect_engagements(utts)
    assert hits[0].has_feedback is False


def test_feedback_outside_window():
    utts = [_u(0, "해보세요"), _u(400, "어떻게 됐어요")]  # 5분(300초) 밖
    hits = eng.detect_engagements(utts)
    assert hits[0].has_feedback is False


# ── 채점 ──────────────────────────────────────────────────────
def test_engagement_score_bands():
    assert eng.engagement_score(3) == 3
    assert eng.engagement_score(1) == 2
    assert eng.engagement_score(0) == 1


def test_bonuses_and_final():
    assert eng.gap_bonus(20) == 1 and eng.gap_bonus(10) == 0
    assert eng.feedback_bonus(2) == 1 and eng.feedback_bonus(0) == 0
    # 유도 2회↑(3) + gap(+1) + feedback(+1) = 5
    assert eng.final_score(3, 20, 1) == 5
    # 유도 0회(1) + 갭 부족 + 피드백 없음 = 1
    assert eng.final_score(0, 5, 0) == 1


def test_profile_keys():
    utts = [_u(0, "한번 해보세요"), _u(45, "자 어떻게 됐어요"), _u(60, "짜보세요"), _u(80, "다음")]
    prof = eng.engagement_profile(utts)
    assert {"engagement_count", "avg_gap", "feedback_count", "final_score",
            "genuine_handoff_count", "genuine_ratio", "baseline_gap_median"} <= set(prof)
    assert prof["engagement_count"] == 2

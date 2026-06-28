"""
preprocessing/item02_completeness.py — 항목 2 발화 완결성: completeness_rate + 1~5 채점

[이수민 - 2026-06-22]
문장화(sentencizer)는 공용 인프라로 두고, 항목 2 채점 로직만 분리한 모듈.
입력은 sentencizer.build_sentences 가 만든 list[Sentence] (Kiwi EF/EC 판정 포함).

  completeness_rate  = 완결 문장 수 / 판정 문장 수 × 100   # 완결 = 마지막 실질 형태소 EF
  completeness_score = completeness_rate → 1~5 (잠정 밴드)
"""
from __future__ import annotations

from app.analysis.schemas import Sentence


# ── 항목 2 지표 씨앗: 완결 문장 비율 ──────────────────────────
def completeness_rate(sentences: list[Sentence]) -> float:
    """완결 문장 비율(%) — 항목 2(발화 완결성) 지표의 씨앗.

    완결 = is_complete True. is_complete None(미판정)은 분모에서 제외.
    """
    judged = [s for s in sentences if s.is_complete is not None]
    if not judged:
        return 0.0
    complete = sum(1 for s in judged if s.is_complete)
    return complete / len(judged) * 100


# ── 항목 2 채점 (잠정 밴드) ────────────────────────────────────
# [이수민 - 2026-06-22] 잠정 밴드. [2026-06-22 재보정] Mecab→Kiwi 전환 반영.
# 형태소 분석기를 Kiwi 로 바꾸자 완결률이 69~72%(Mecab) → 96.5%(Kiwi, σ0.4)로 상승.
#   원인: Mecab 이 구어 종결어미(게요/냐/죠/어요/구나/야 등)를 EC 로 오태깅해 완결을
#   과소집계하던 것을 Kiwi 가 바로잡음(15강의 26%가 Mecab EF→Kiwi EF, 검수 결과 Kiwi 정답).
# 근거 부재 주의: 출판 컷오프 없음. 아래는 Kiwi 실측(평균 96.5%)을 '거의 완전(5)'에 두고
#   trailing-off(EC 종결)가 늘수록 감점하는 **내부 잠정 기준**(출판 근거 아님).
#   주의: 정확 태깅 하에선 유창한 강사의 완결률이 포화(≈96%)라 변별력은 약함 — 절대 등급 용도.
def completeness_score(rate: float) -> int:
    """completeness_rate(%) → 1~5 (잠정 밴드, Kiwi 분포 기준). 상세는 위 주석/문서."""
    if rate >= 93:
        return 5
    if rate >= 87:
        return 4
    if rate >= 78:
        return 3
    if rate >= 68:
        return 2
    return 1

"""
preprocessing/formality.py — 항목 3 언어 일관성: 존댓말/반말/중립 라벨링 + consistency_ratio

[이수민 - 2026-06-15]
sentencizer 가 만든 Sentence 의 '마지막 EF 형태소'를 받아 말투를 분류한다.
항목 2(완결성)와 같은 형태소를 공유 — 2는 "EF냐(완결)", 3은 "그 EF가 존댓말이냐".

분류 (Mecab 종결어미 표면형 기반, 실데이터 분포로 설계):
  격식존댓말   : ~니다 / ~니까        (합니다, 습니다, 됩니다, 습니까)
  비격식존댓말 : ~요 / ~죠            (어요, 죠, 예요, 세요, 거든요, 돼요, 잖아요 …)
  반말         : ~야/어/아/지/잖아/거든/을까 …  (요·니다·다 로 안 끝남)
  중립         : ~다(한다체 평서) / EF 아님(명사 종결·불완결) → 분모에서 제외

[데이터 발견] '-다'(한다/된다/것이다/싶다)는 진짜 반말이 아니라 '한다체' 중립 서술.
  3개 파일 실측: '야/어/잖아'는 진짜 반말로 확인됐으나 '-다'는 "하시면 된다",
  "의미한다" 처럼 개념 설명용 평서. 반말로 세면 일관성이 폭락(EF의 ~14%가 '-다')
  → 중립으로 분리. (루브릭 분류표는 '-다(구어)'를 반말로 뒀으나 EDA 결과 한다체가
  대다수 → 중립 처리 제안. 채점 밴드 확정 시 팀 협의.)

지표 (루브릭 항목 3):
  존댓말 = 격식 + 비격식
  consistency_ratio = max(존댓말 비율, 반말 비율)   # 분모 = 존댓말 + 반말 (중립 제외)
  violation_count   = (존댓말 + 반말) - 지배 말투 문장 수
임계값 상태: ⚠️ EDA 후 수정 — consistency_ratio 분포로 채점 밴드 확정 예정.
"""
from __future__ import annotations

from collections import Counter

from app.analysis.schemas import Sentence

# 말투 라벨
FORMAL = "격식존댓말"
INFORMAL_POLITE = "비격식존댓말"
BANMAL = "반말"
NEUTRAL = "중립"

_POLITE_SUFFIX = ("요", "죠")          # 해요체 (비격식 존댓말)
_FORMAL_SUFFIX = ("니다", "니까")      # 하십시오체 (격식 존댓말)
_DECLARATIVE_SUFFIX = ("다",)          # 한다체 평서 (중립) — 한다/된다/것이다/싶다


def classify_formality(ending_morph: str | None, ending_tag: str | None) -> str:
    """문장 마지막 EF 형태소 표면형 → 말투 라벨.

    EF 가 아니면(불완결·명사 종결) 중립. 표면형 접미 패턴으로 분류.
    순서 주의: 요/죠(존댓말) → 니다/니까(격식) → 다(한다체 중립) → 나머지(반말).
    """
    if not ending_tag or "EF" not in ending_tag or not ending_morph:
        return NEUTRAL
    m = ending_morph
    if m.endswith(_POLITE_SUFFIX):
        return INFORMAL_POLITE
    if m.endswith(_FORMAL_SUFFIX):
        return FORMAL
    if m.endswith(_DECLARATIVE_SUFFIX):
        return NEUTRAL              # 한다체 평서 → 중립
    return BANMAL


# ── 항목 3 채점 (잠정 밴드) ────────────────────────────────────
# [이수민 - 2026-06-22] 잠정 score band 추가.
# 근거 부재 주의: 일관성 %의 출판 컷오프 기준치는 없음. 아래는 "한 말투로 일관될수록
#   고점"이라는 단순 원칙의 **내부 잠정 기준**(출판 근거 아님). 실측(15강의 67~72%,
#   존댓말/반말 ~72/28 혼용)은 2~3점대에 위치 — 혼용이 적지 않다는 판단. 박하면 컷오프 하향.
# violation_count 미반영: 현재 consistency_ratio 만으로 채점(위반 문장 수 가중 감점은 보류).
# 적용 범위: 팀 안내(2026-06-22) — 단일 강사만 대상. 변별 보정 불요. 절대 등급 + 주차 trend 용도.
# 컷오프/violation 반영 확정 시 docs/scoring-bands.md 갱신 + 팀 협의.
def consistency_score(ratio: float) -> int:
    """consistency_ratio(%) → 1~5 (잠정 밴드). 상세 근거는 위 주석/문서 참고."""
    if ratio >= 90:
        return 5
    if ratio >= 80:
        return 4
    if ratio >= 70:
        return 3
    if ratio >= 60:
        return 2
    return 1


def formality_profile(sentences: list[Sentence]) -> dict:
    """문장 리스트 말투 집계 + consistency_ratio / violation_count.

    반환 키:
        counts             라벨별 문장 수 (격식/비격식/반말/중립)
        jondaetmal/banmal/neutral  묶음 수 (존댓말 = 격식 + 비격식)
        jondaetmal_ratio/banmal_ratio  중립 제외 분모 기준 비율(%)
        consistency_ratio  max(존댓말%, 반말%)
        violation_count    비지배 말투 문장 수
        dominant           'jondaetmal' | 'banmal' | None
    """
    counts = Counter(classify_formality(s.ending_morph, s.ending_tag) for s in sentences)
    jond = counts[FORMAL] + counts[INFORMAL_POLITE]
    banmal = counts[BANMAL]
    neutral = counts[NEUTRAL]
    denom = jond + banmal
    base = {
        "counts": {FORMAL: counts[FORMAL], INFORMAL_POLITE: counts[INFORMAL_POLITE],
                   BANMAL: counts[BANMAL], NEUTRAL: counts[NEUTRAL]},
        "jondaetmal": jond, "banmal": banmal, "neutral": neutral,
    }
    if denom == 0:
        return {**base, "jondaetmal_ratio": 0.0, "banmal_ratio": 0.0,
                "consistency_ratio": 0.0, "consistency_score": consistency_score(0.0),
                "violation_count": 0, "dominant": None}
    jond_ratio, banmal_ratio = jond / denom * 100, banmal / denom * 100
    cratio = max(jond_ratio, banmal_ratio)
    return {
        **base,
        "jondaetmal_ratio": jond_ratio,
        "banmal_ratio": banmal_ratio,
        "consistency_ratio": cratio,
        "consistency_score": consistency_score(cratio),   # 잠정 밴드
        "violation_count": denom - max(jond, banmal),
        "dominant": "jondaetmal" if jond >= banmal else "banmal",
    }

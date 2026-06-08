"""BoW 행동 태깅 (2-A단계) — Paper #3.

모든 발화 행을 순회하며 18개 항목별 긍정/부정 지표 출현을 카운트.
``configs/bow_indicators.yaml`` 의 키워드 사전을 사용.

LLM이 데이터가 적을 때 흔들리는 신뢰도를 BoW가 보조한다.

TODO:
- 키워드 사전 로더
- 정규식 기반 매칭 (어절 경계 인식)
- "동일 단어 3회 연속" 같은 규칙형 지표
- raw_count → 1~5 정규화 (configs.normalization)
"""
from app.analysis.schemas import BehaviorProfile, LectureDocument


def tag(document: LectureDocument) -> BehaviorProfile:
    """LectureDocument → 18 항목 BoW 프로필 (placeholder)."""
    raise NotImplementedError


def normalize_bow_score(positive: int, negative: int, scale: float = 5.0) -> float:
    """raw count → 1~5 정규화 점수 (placeholder).

    근거 부족 시 0 반환 (ensemble 에서 LLM 점수만 사용하도록).
    """
    raise NotImplementedError

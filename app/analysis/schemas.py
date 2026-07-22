"""v2 데이터 스키마.

파이프라인의 각 단계에서 주고받는 Pydantic 모델 정의.

흐름:
    raw_text
      → LectureDocument          (preprocessor)
      → BehaviorProfile          (behavior_tagger, 18항목 BoW)
      → LectureIndex             (embedder, 청크 + 임베딩)
      → ItemAnalysis x 18        (analyzer, LLM raw)
      → ItemScore x 18           (ensemble, 최종 점수)
      → InstructorScorecard      (scorer, 카테고리 + 트렌드)

[이수민 - 2026-06-15]
Sentence 모델 추가 — STT 라인 파편을 재구성한 '문장' 단위.
문장화(preprocessing/sentencizer.py): 갭 기반 발화 세그먼트 분리 → Kiwi 문장 분리
→ Kiwi EF/EC 완결성 판정. 항목 2(발화 완결성)·3(언어 일관성)의 입력 단위.
LectureDocument 에 편입 — build_document(with_sentences=True) 시 sentences/
completeness_rate/consistency_ratio/violation_count 채워짐. 점수화(1~5)는 스코어러 몫.
"""
from datetime import datetime

from pydantic import BaseModel, Field


# ─── 1단계: 전처리 산출물 ────────────────────────────────────
class Utterance(BaseModel):
    """단일 발화 행: `<HH:MM:SS> speaker_id: text`"""

    timestamp: str               # "09:11:17"
    speaker_id: str              # "b54f46b0"
    text: str
    seconds_from_start: int      # 강의 시작 후 경과 초


class Session(BaseModel):
    """한 세션(예: 오전/오후) 단위 묶음.

    한 파일에 오전·오후 세션이 합쳐져 있으므로 30분 이상 갭으로 분리한다.
    """

    session_index: int                # 0, 1, ...
    label: str                        # "morning" / "afternoon" / "session_N"
    start_timestamp: str
    end_timestamp: str
    all_lines: list[Utterance]
    intro_lines: list[Utterance]      # 세션 시작 후 N분
    middle_lines: list[Utterance]
    outro_lines: list[Utterance]      # 세션 종료 전 M분


# ─── 1단계(보강): 문장 재구성 산출물 (LectureDocument 이전에 정의) ──
class Sentence(BaseModel):
    """STT 라인 파편을 재구성한 '문장' 단위 (sentencizer 산출).

    STT 는 `<HH:MM:SS> id: text` 라인이라 문장이 아니다. 갭 기반 세그먼트 분리 →
    Kiwi 문장 분리 → Kiwi EF/EC 판정으로 만든다. 항목 2·3 의 입력.
    """

    text: str
    segment_index: int                 # 소속 발화 세그먼트 인덱스
    start_timestamp: str               # 문장 첫 라인 타임스탬프
    end_timestamp: str                 # 문장 마지막 라인 타임스탬프
    start_seconds: int                 # 첫 라인 경과초
    end_seconds: int                   # 마지막 라인 경과초
    utterance_indices: list[int] = Field(default_factory=list)  # 원본 all_lines 인덱스
    ends_with_gap: bool = False        # 세그먼트 끝(뒤 갭 > 임계값) → 발화 끊김 후보
    ending_morph: str | None = None    # 마지막 실질 형태소 표면형
    ending_tag: str | None = None      # 마지막 실질 형태소 품사 (EF/EC/...)
    is_complete: bool | None = None    # EF=True, EC 등=False, 미판정(use_morph off)=None


class LectureDocument(BaseModel):
    """전처리된 강의 한 편."""

    lecture_date: str            # "2026-02-02"
    instructor_id: str
    all_lines: list[Utterance]
    # 세션 단위 분리 (오전/오후) — discrete 항목(4,5,8 등) 평가용
    sessions: list[Session] = Field(default_factory=list)
    # 파일 평탄화 — 기존 다운스트림(embedder/analyzer) 호환
    intro_lines: list[Utterance] = Field(default_factory=list)
    middle_lines: list[Utterance] = Field(default_factory=list)
    outro_lines: list[Utterance] = Field(default_factory=list)
    # 보조 통계 (BoW 입력)
    filler_word_ratio: float = 0.0    # 추임새 비율
    avg_line_gap_seconds: float = 0.0 # 발화 간 평균 간격
    # ── 항목 2·3 원지표 + 공통 입력 (with_sentences=True 시 채워짐) ──
    # 점수(1~5) 아님 — 밴드 적용은 스코어러/통합 담당 몫.
    sentences: list[Sentence] = Field(default_factory=list)   # 항목 2·3·9·10 공통 입력
    completeness_rate: float = 0.0    # 항목 2 원지표 (완결 문장 %)
    consistency_ratio: float = 0.0    # 항목 3 원지표 (지배 말투 %)
    violation_count: int = 0          # 항목 3 원지표 (비지배 말투 문장 수)


# ─── 2-A단계: BoW 행동 태깅 산출물 (미사용 — 참조 0건, behavior_tagger.py 제거됨) ──
# class ItemBoW(BaseModel):
#     """단일 항목의 BoW 집계."""
#
#     positive_count: int = 0
#     negative_count: int = 0
#     bow_score: float = 0.0       # 1~5 정규화, 0이면 근거 없음
#
#
# class BehaviorProfile(BaseModel):
#     """18개 항목의 BoW 결과 묶음."""
#
#     lecture_date: str
#     instructor_id: str
#     items: dict[int, ItemBoW]    # {item_id: ItemBoW}


# ─── 2-B단계: RAG 인덱싱 산출물 (미사용 — 참조 0건, embedder.py 경로 미사용) ──
# class IndexedChunk(BaseModel):
#     """임베딩된 청크 (15행 기본)."""
#
#     chunk_id: int
#     start_timestamp: str
#     end_timestamp: str
#     text: str
#     embedding: list[float] | None = None   # 캐싱 시 None 가능
#     line_indices: list[int]                # 원본 all_lines 인덱스
#
#
# class LectureIndex(BaseModel):
#     """강의 1편 분량 벡터 인덱스."""
#
#     lecture_date: str
#     chunks: list[IndexedChunk]
#
#     class Config:
#         # 임베딩 배열이 크니 직렬화 시 주의
#         arbitrary_types_allowed = True


# ─── 3단계: LLM 분석 산출물 (raw) (미사용 — 참조 0건, ensemble.py 제거됨) ──
# class LLMItemRaw(BaseModel):
#     """LLM 원본 JSON 응답 (단일 항목)."""
#
#     item_id: int
#     score: float = Field(ge=1.0, le=5.0)
#     evidence: str                # 근거 인용 (원문 발췌)
#     strengths: str
#     improvements: str
#     confidence: float = Field(ge=0.0, le=1.0)
#     used_chunk_ids: list[int] = Field(default_factory=list)  # RAG 추적


# ─── 4단계: 앙상블 산출물 (final) ────────────────────────────
class ItemScore(BaseModel):
    """앙상블된 최종 항목 점수."""

    item_id: int
    name: str
    category: str
    item_type: str               # "discrete" | "high_inference"
    final_score: float = Field(ge=1.0, le=5.0)
    llm_score: float
    bow_score: float
    final_confidence: float
    evidence: str
    reason: str = ""             # 점수 근거 해설 (왜 이 점수인지) — 항목표 '해설' 칸용
    strengths: str
    improvements: str
    needs_human_review: bool = False    # confidence < threshold 시 True


# ─── 5단계: 스코어링 산출물 ──────────────────────────────────
class CategoryScore(BaseModel):
    category: str
    score: float                 # 카테고리 내 평균
    weight: float                # 종합 점수 가중치


class TrendPoint(BaseModel):
    date: str
    score: float


class InstructorScorecard(BaseModel):
    """강사 1명의 종합 분석 결과."""

    instructor_id: str
    lecture_date: str
    overall_score: float                 # 1~5
    category_scores: list[CategoryScore]
    item_scores: list[ItemScore]         # 18개
    # 시계열 (다중 강의 분석 시 채워짐)
    trend_slope: float | None = None     # 선형 회귀 기울기
    trend_label: str | None = None       # "improving" | "stable" | "declining"
    trend_points: list[TrendPoint] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=datetime.now)

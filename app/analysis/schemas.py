"""v2 데이터 스키마.

파이프라인의 각 단계에서 주고받는 Pydantic 모델 정의.

흐름:
    raw_text
      → LectureDocument          (preprocessor)
      → BehaviorProfile          (behavior_tagger, 18항목 BoW)
      → LectureIndex             (embedder, 청크 + 임베딩)
      → ItemAnalysis × 18        (analyzer, LLM raw)
      → ItemScore × 18           (ensemble, 최종 점수)
      → InstructorScorecard      (scorer, 카테고리 + 트렌드)
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


class LectureDocument(BaseModel):
    """전처리된 강의 한 편."""

    lecture_date: str            # "2026-02-02"
    instructor_id: str
    all_lines: list[Utterance]
    intro_lines: list[Utterance]      # 시작 후 20분
    middle_lines: list[Utterance]     # 중간
    outro_lines: list[Utterance]      # 종료 전 15분
    # 보조 통계 (BoW 입력)
    filler_word_ratio: float = 0.0    # 추임새 비율
    avg_line_gap_seconds: float = 0.0 # 발화 간 평균 간격


# ─── 2-A단계: BoW 행동 태깅 산출물 ───────────────────────────
class ItemBoW(BaseModel):
    """단일 항목의 BoW 집계."""

    positive_count: int = 0
    negative_count: int = 0
    bow_score: float = 0.0       # 1~5 정규화, 0이면 근거 없음


class BehaviorProfile(BaseModel):
    """18개 항목의 BoW 결과 묶음."""

    lecture_date: str
    instructor_id: str
    items: dict[int, ItemBoW]    # {item_id: ItemBoW}


# ─── 2-B단계: RAG 인덱싱 산출물 ──────────────────────────────
class IndexedChunk(BaseModel):
    """임베딩된 청크 (15행 기본)."""

    chunk_id: int
    start_timestamp: str
    end_timestamp: str
    text: str
    embedding: list[float] | None = None   # 캐싱 시 None 가능
    line_indices: list[int]                # 원본 all_lines 인덱스


class LectureIndex(BaseModel):
    """강의 1편 분량 벡터 인덱스."""

    lecture_date: str
    chunks: list[IndexedChunk]

    class Config:
        # 임베딩 배열이 크니 직렬화 시 주의
        arbitrary_types_allowed = True


# ─── 3단계: LLM 분석 산출물 (raw) ────────────────────────────
class LLMItemRaw(BaseModel):
    """LLM 원본 JSON 응답 (단일 항목)."""

    item_id: int
    score: float = Field(ge=1.0, le=5.0)
    evidence: str                # 근거 인용 (원문 발췌)
    strengths: str
    improvements: str
    confidence: float = Field(ge=0.0, le=1.0)
    used_chunk_ids: list[int] = Field(default_factory=list)  # RAG 추적


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

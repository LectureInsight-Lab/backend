"""분석 API 라우트 (v2).

파이프라인 진입점:
    POST /api/v1/analysis/lecture            단일 강의 1편 분석
    POST /api/v1/analysis/batch              다중 강의 배치 분석 (시계열 트렌드용)
    GET  /api/v1/analysis/instructor/{id}    강사별 누적 분석 결과 조회
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.analysis import pipeline, scorer
from app.analysis.schemas import InstructorScorecard
from app.core import store
from app.core.paths import read_stt

router = APIRouter()


class LectureRequest(BaseModel):
    """단일 강의 분석 요청.

    ``raw_text`` 를 직접 주거나, ``course_id`` 를 주면 configs/paths.yaml 의
    stt_dir 에서 ``{date}_{course_id}.txt`` 를 읽는다.
    """

    instructor_id: str
    lecture_date: str = Field(..., description="YYYY-MM-DD")
    raw_text: str | None = None
    course_id: str | None = None

    @model_validator(mode="after")
    def _need_source(self):
        if not self.raw_text and not self.course_id:
            raise ValueError("raw_text 또는 course_id 중 하나는 필요합니다.")
        return self


class BatchRequest(BaseModel):
    lectures: list[LectureRequest]


async def _run_one(req: LectureRequest) -> InstructorScorecard:
    raw_text = req.raw_text
    if raw_text is None:
        try:
            raw_text = read_stt(req.lecture_date, req.course_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    card = await pipeline.analyze_raw_text(raw_text, req.lecture_date, req.instructor_id)
    store.save(card)                                    # 1차 저장
    history = store.list_by_instructor(req.instructor_id)
    scorer.attach_trend(card, history)                  # 누적 이력으로 트렌드 부착
    store.save(card)                                    # 트렌드 반영 재저장
    return card


@router.post("/lecture", response_model=InstructorScorecard)
async def analyze_lecture(req: LectureRequest) -> InstructorScorecard:
    """단일 강의 분석.

    파이프라인: preprocessor → behavior_tagger + embedder → analyzer (LLM × 18 병렬)
                → ensemble → scorer. 응답: InstructorScorecard.
    """
    return await _run_one(req)


@router.post("/batch")
async def analyze_batch(req: BatchRequest) -> dict:
    """다중 강의 배치 분석 (강사별 시계열 산출). 응답: scorecards + 주차 집계."""
    cards = [await _run_one(lec) for lec in req.lectures]
    return {
        "scorecards": [c.model_dump() for c in cards],
        "weekly": scorer.aggregate_weekly(cards),
    }


@router.get("/instructor/{instructor_id}")
async def get_instructor_history(instructor_id: str) -> dict:
    """강사별 누적 분석 결과 + 트렌드 + 주차 집계."""
    cards = store.list_by_instructor(instructor_id)
    if not cards:
        raise HTTPException(status_code=404, detail=f"분석 이력 없음: {instructor_id}")
    slope, label = scorer.compute_trend(
        [scorer.TrendPoint(date=c.lecture_date, score=c.overall_score) for c in cards]
    )
    return {
        "instructor_id": instructor_id,
        "scorecards": [c.model_dump() for c in cards],
        "trend_slope": slope,
        "trend_label": label,
        "weekly": scorer.aggregate_weekly(cards),
    }

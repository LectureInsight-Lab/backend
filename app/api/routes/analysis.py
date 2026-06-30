"""분석 API 라우트 (v2).

파이프라인 진입점:
    POST /api/v1/analysis/lecture            단일 강의 1편 분석
    POST /api/v1/analysis/batch              다중 강의 배치 분석 (시계열 트렌드용)
    GET  /api/v1/analysis/instructor/{id}    강사별 누적 분석 결과 조회
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from app.analysis import pipeline, scorer
from app.analysis.schemas import InstructorScorecard
from app.core import jobs, store
from app.core.paths import read_stt

router = APIRouter()

# 백그라운드 작업 참조 보관 (GC 로 사라지는 것 방지)
_bg_tasks: set[asyncio.Task] = set()


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


class NarrativeRequest(BaseModel):
    """종합 자연어 분석 요청. 단일/종합 스코어카드를 받아 해설+요약을 생성."""

    scorecard: InstructorScorecard
    is_aggregate: bool = False      # 입력 파일 종합(여러 강의 평균) 뷰면 True → 종합 톤
    lecture_count: int = 1


@router.post("/narrative")
async def analysis_narrative(req: NarrativeRequest) -> dict:
    """스코어카드 → {overall_feedback, summary}.

    (1) 종합 자연어 해설 생성 → (2) 그 해설을 요약. is_aggregate 에 따라 톤이 바뀐다.
    프론트가 현재 보는 카드(단일 일자 / 종합)를 그대로 보내 호출한다.
    """
    from app.analysis import narrative

    return await narrative.generate_narrative(
        req.scorecard, req.is_aggregate, req.lecture_count
    )


async def _run_one(req: LectureRequest) -> InstructorScorecard:
    logger.info(
        f"[api] 분석 요청 — instructor={req.instructor_id} date={req.lecture_date} "
        f"source={'raw_text' if req.raw_text else f'course={req.course_id}'}"
    )
    raw_text = req.raw_text
    if raw_text is None:
        try:
            raw_text = read_stt(req.lecture_date, req.course_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    logger.info(f"[api] 원문 길이 {len(raw_text)}자 → 파이프라인 진입")
    card = await pipeline.analyze_raw_text(raw_text, req.lecture_date, req.instructor_id)
    store.save(card)                                    # 1차 저장
    history = store.list_by_instructor(req.instructor_id)
    scorer.attach_trend(card, history)                  # 누적 이력으로 트렌드 부착
    store.save(card)                                    # 트렌드 반영 재저장
    logger.success(f"[api] ✓ 완료 — overall={card.overall_score}")
    return card


@router.post("/lecture", response_model=InstructorScorecard)
async def analyze_lecture(req: LectureRequest) -> InstructorScorecard:
    """단일 강의 분석.

    파이프라인: preprocessor → behavior_tagger + embedder → analyzer (LLM × 18 병렬)
                → ensemble → scorer. 응답: InstructorScorecard.
    """
    return await _run_one(req)


class JobStartResponse(BaseModel):
    job_id: str


async def _run_job(job_id: str, req: LectureRequest) -> None:
    """백그라운드 분석 1건 — 진행률을 job 저장소에 보고하며 실행."""
    sink = jobs.JobProgressSink(job_id)
    try:
        jobs.mark_running(job_id)
        raw_text = req.raw_text
        if raw_text is None:
            raw_text = read_stt(req.lecture_date, req.course_id)  # FileNotFoundError 가능
        card = await pipeline.analyze_raw_text(
            raw_text, req.lecture_date, req.instructor_id, progress=sink
        )
        store.save(card)
        history = store.list_by_instructor(req.instructor_id)
        scorer.attach_trend(card, history)
        store.save(card)
        jobs.finish_job(job_id, card.model_dump(mode="json"))
        logger.success(f"[api] ✓ job {job_id} 완료 — overall={card.overall_score}")
    except Exception as e:  # noqa: BLE001 — 어떤 실패든 job 에 기록
        logger.exception(f"[api] ✗ job {job_id} 실패")
        jobs.fail_job(job_id, f"{type(e).__name__}: {e}")


@router.post("/lecture/async", response_model=JobStartResponse)
async def analyze_lecture_async(req: LectureRequest) -> JobStartResponse:
    """단일 강의 분석을 백그라운드로 시작하고 즉시 job_id 를 반환한다.

    진행률/로그는 ``GET /api/v1/analysis/job/{job_id}`` 로 폴링한다.
    """
    job_id = jobs.create_job(label=f"{req.instructor_id} · {req.lecture_date}")
    task = asyncio.create_task(_run_job(job_id, req))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return JobStartResponse(job_id=job_id)


@router.get("/job/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """분석 작업 진행 상태 조회 (status/percent/stage/logs/result)."""
    snap = jobs.get_job(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"job 없음: {job_id}")
    return snap


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

"""분석 API 라우트 (v2).

파이프라인 진입점:
    POST /api/v1/analysis/lecture            단일 강의 1편 분석
    POST /api/v1/analysis/batch              다중 강의 배치 분석 (시계열 트렌드용)
    GET  /api/v1/analysis/instructor/{id}    강사별 누적 분석 결과 조회
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/lecture")
async def analyze_lecture():
    """단일 강의 분석.

    파이프라인: preprocessor → behavior_tagger + embedder
                → analyzer (LLM × 18 병렬)
                → ensemble → scorer
    응답: InstructorScorecard
    """
    raise NotImplementedError


@router.post("/batch")
async def analyze_batch():
    """다중 강의 배치 분석 (강사별 시계열 산출 용).

    응답: list[InstructorScorecard] + trend
    """
    raise NotImplementedError


@router.get("/instructor/{instructor_id}")
async def get_instructor_history(instructor_id: str):
    """강사별 누적 분석 결과 (DB 영속화 필요)."""
    raise NotImplementedError

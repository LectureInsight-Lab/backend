from fastapi import APIRouter

router = APIRouter()


@router.post("/lecture")
async def analyze_lecture():
    """단일 강의 스크립트 분석 (placeholder)."""
    raise NotImplementedError


@router.post("/batch")
async def analyze_batch():
    """다중 강의 배치 분석 (placeholder)."""
    raise NotImplementedError


@router.get("/instructor/{instructor_id}")
async def get_instructor_analysis(instructor_id: str):
    """강사별 누적 분석 결과 (placeholder)."""
    raise NotImplementedError

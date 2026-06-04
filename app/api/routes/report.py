from fastapi import APIRouter

router = APIRouter()


@router.post("/generate")
async def generate_report():
    """분석 결과를 PDF/DOCX 리포트로 생성 (placeholder)."""
    raise NotImplementedError


@router.get("/compare")
async def compare_instructors():
    """강사 간 비교 리포트 (placeholder)."""
    raise NotImplementedError

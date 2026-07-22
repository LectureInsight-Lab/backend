from urllib.request import Request

from fastapi import APIRouter

from app.schemas.analysis import AnalysisRequest
from app.service.analysis_service import analyze_lecture

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
)

@router.post("")
async def analyze_lecture(request: AnalysisRequest):
    return await analyze_lecture(request)
"""리포트 API 라우트 (v2)."""
from fastapi import APIRouter

router = APIRouter()


@router.post("/generate")
async def generate_report():
    """InstructorScorecard → HTML/DOCX 리포트 생성.

    요청 body 예::
        {"scorecard_id": "...", "formats": ["html", "docx"]}
    응답: {"html": "outputs/...", "docx": "outputs/..."}
    """
    raise NotImplementedError


@router.get("/download/{file_id}")
async def download_report(file_id: str):
    """생성된 리포트 다운로드 (placeholder)."""
    raise NotImplementedError

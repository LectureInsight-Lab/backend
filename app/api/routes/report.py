from fastapi import APIRouter

from app.schemas.report import ReportRequest
from app.service.report_service import generate_report

router = APIRouter(
    prefix="/report",
    tags=["report"],
)

@router.post("")
def generate_report(request: ReportRequest):
    return generate_report(request)
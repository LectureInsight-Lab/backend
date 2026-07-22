from pydantic import BaseModel
from typing import Literal

class ReportRequest(BaseModel):
    scorecard_id: str
    formats: list[Literal["html", "docx"]] = ["html", "docx"]
from pydantic import BaseModel, ConfigDict, Field
from datetime import date
from typing import Any

class AnalysisRequest(BaseModel):
    instructor_id: str
    lecture_date: str
    raw_text: str | None = None
    course_id: str | None = None

class AnalysisResponse(BaseModel):
    scorecard_id: str
    instructor_id: str
    lecture_date: str

    scores: dict[str, Any] = Field(
        description="강의 평가 항목별 점수"
    )
    trend: dict[str, Any] = Field(
        default=None,
        description="과거 강의 분석 결과와 비교한 추세"
    )
    narrative: dict[str, Any] | None = Field(
        default=None,
        description="종합 해설과 요약"
    )
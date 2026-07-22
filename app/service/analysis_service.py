from app.core.paths import read_stt
from app.schemas.analysis import AnalysisRequest
from app.analysis.pipeline import analyze_raw_text
from app.core.store import save, list_by_instructor
from app.analysis.scorer import attach_trend

async def analyze_lecture(request: AnalysisRequest):
    raw_text = request.raw_text

    if raw_text is None:
        if request.course_id is None:
            raise ValueError("course_id 또는 text가 필요합니다.")

        raw_text = read_stt(
            lecture_date=request.lecture_date,
            course_id=request.course_id
        )

    scorecard =  await analyze_raw_text(raw_text, request.lecture_date, request.instructor_id)

    history = list_by_instructor(request.instructor_id)

    attach_trend(scorecard,history)

    save(scorecard)

    return scorecard


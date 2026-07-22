from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.analysis import AnalysisRequest
from app.service.analysis_service import analyze_lecture
import pytest

client = TestClient(app)


def test_analyze_lecture_api(mocker):
    fake_result = {
        "instructor_id": "teacher01",
        "lecture_date": "2026-07-21",
        "overall_score": 4.2,
        "item_scores": [],
    }

    mocked_service = mocker.patch(
        "app.api.routes.analysis.analyze_lecture",
        new=AsyncMock(return_value=fake_result),
    )

    request_body = {
        "instructor_id": "teacher01",
        "lecture_date": "2026-07-21",
        "raw_text": "테스트용 강의 원문입니다.",
        "course_id": None,
    }

    response = client.post(
        "/api/v1/analysis",
        json=request_body,
    )

    assert response.status_code == 200
    assert response.json() == fake_result

    mocked_service.assert_awaited_once()




@pytest.mark.asyncio
async def test_analyze_lecture_with_raw_text(mocker):
    fake_scorecard = Mock()
    fake_history = [Mock(), Mock()]

    mocked_analyze = mocker.patch(
        "app.service.analysis_service.analyze_raw_text",
        new=AsyncMock(return_value=fake_scorecard),
    )
    mocked_history = mocker.patch(
        "app.service.analysis_service.list_by_instructor",
        return_value=fake_history,
    )
    mocked_trend = mocker.patch(
        "app.service.analysis_service.attach_trend",
    )
    mocked_save = mocker.patch(
        "app.service.analysis_service.save",
    )
    mocked_read_stt = mocker.patch(
        "app.service.analysis_service.read_stt",
    )

    request = AnalysisRequest(
        instructor_id="teacher01",
        lecture_date="2026-07-21",
        raw_text="테스트용 강의 원문",
        course_id=None,
    )

    result = await analyze_lecture(request)

    assert result is fake_scorecard

    mocked_read_stt.assert_not_called()

    mocked_analyze.assert_awaited_once_with(
        "테스트용 강의 원문",
        "2026-07-21",
        "teacher01",
    )

    mocked_history.assert_called_once_with("teacher01")
    mocked_trend.assert_called_once_with(fake_scorecard, fake_history)
    mocked_save.assert_called_once_with(fake_scorecard)
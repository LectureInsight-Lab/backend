from unittest.mock import Mock
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.report import ReportRequest
from app.service.report_service import generate_report

client = TestClient(app)


def test_create_report_api(mocker):
    fake_result = {
        "html": "outputs/teacher01/2026-07-21.html",
        "docx": "outputs/teacher01/2026-07-21.docx",
    }

    mocked_service = mocker.patch(
        "app.api.routes.report.generate_report",
        return_value=fake_result,
    )

    request_body = {
        "scorecard_id": "1",
        "formats": ["html", "docx"],
    }

    response = client.post(
        "/api/v1/report",
        json=request_body,
    )

    assert response.status_code == 200
    assert response.json() == fake_result

    mocked_service.assert_called_once()


def test_generate_report(mocker):
    fake_scorecard = Mock()

    fake_result = {
        "html": "outputs/teacher01/2026-07-21.html",
        "docx": "outputs/teacher01/2026-07-21.docx",
    }

    mocked_get = mocker.patch(
        "app.service.report_service.get",
        return_value=fake_scorecard,
    )

    mocked_generate = mocker.patch(
        "app.service.report_service.generate",
        return_value=fake_result,
    )

    request = ReportRequest(
        scorecard_id="1",
        formats=["html", "docx"],
    )

    result = generate_report(request)

    assert result == fake_result

    mocked_get.assert_called_once_with("1")

    mocked_generate.assert_called_once_with(
        scorecard=fake_scorecard,
        formats=["html", "docx"],
    )

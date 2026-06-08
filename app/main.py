"""FastAPI 진입점 (v2).

분석/리포트 REST API. Streamlit 대시보드(``app/dashboard/app.py``)는
이 API를 호출해 결과를 조회한다.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analysis, health, report
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="LectureInsight API",
        description=(
            "강의 스크립트 분석 및 강사 리포트 생성 API (v2). "
            "RAG + BoW 앙상블 기반 18개 항목 평가."
        ),
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
    app.include_router(report.router, prefix="/api/v1/report", tags=["report"])

    return app


app = create_app()

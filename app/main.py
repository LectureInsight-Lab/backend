"""FastAPI 진입점 (v2).

분석/리포트 REST API. 프론트엔드(Next.js, ``frontend/``)가 이 API를 호출해
분석을 실행하고 결과를 조회한다.
"""
from fastapi import FastAPI
from app.api.routes import analysis, health, report
from app.core.config import settings




def create_app() -> FastAPI:
    app = FastAPI(
        title="LectureInsight API",
        description=("강의 스크립트 분석 및 강사 리포트 생성 API."),
        version="0.2.0",
    )

    app.include_router(health.router)
    app.include_router(analysis.router, prefix="/api/v1")
    app.include_router(report.router, prefix="/api/v1")

    return app

app = create_app()



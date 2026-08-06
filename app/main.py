"""FastAPI 진입점 (v2).

분석/리포트 REST API. 프론트엔드(Next.js, ``frontend/``)가 이 API를 호출해
분석을 실행하고 결과를 조회한다.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # 교차출처 요청 처리 미들웨어
# 프라우저는 프론트에서 다른 출처의 API를 부르면 기본적으로 막음 -> 이를 풀어주는 도구

from app.api.routes import analysis, health, report # 라우트(엔드포인트 묶음) 모듈 3개를 가져옴
from app.core.config import settings


def create_app() -> FastAPI: # 앱 팩토리 함수 정의. FastAPI 인스턴스 반환
    app = FastAPI( # API 문서(/docs/openapi.json)에 그대로 표시되는 메타 데이터.
        title="LectureInsight API",
        description=(
            "강의 스크립트 분석 및 강사 리포트 생성 API (v2). "
            "RAG + BoW 앙상블 기반 18개 항목 평가."
        ),
        version="0.2.0",
    )

    app.add_middleware( # 모든 요청/응답이 통과하는 공통 관문
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 가져온 라우터 3개를 앱에 등록(연결). 여기서 실제 URL 경로가 조립됨
    app.include_router(health.router, tags=["health"]) # tags = /docs 에서 엔드포인트를 그룹으로 묶어 보여주는 라벨
    app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
    app.include_router(report.router, prefix="/api/v1/report", tags=["report"])

    return app


app = create_app() 
# 팩토리를 실제로 한 번 호출해 모듈 수준 app 변수를 만듦. uvicorn main:app 실행 시 uvicorn이 찾는 게 바로 이 app.
# 즉 서버가 물고 들어오는 최종 진입점.

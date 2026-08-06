"""
실제 분석 실행
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger
from datetime import date

from app.analysis.pipeline import analyze_raw_text
from app.analysis.schemas import InstructorScorecard

router = APIRouter()

# 3. 라우트 ──────────────────────────────────────────
# 입력 데이터 STT 생성
@router.post("/run", response_model=InstructorScorecard)
async def analyze(
        file: UploadFile = File(...),
        lecture_date: date = Form(...),  # 값 형식 검증 필요할까?
        instructor_id: str = Form(...),
):
    # txt 확장자인지 확인
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=400, # Bad Request
            detail=f"'.txt' 파일만 지원합니다. 업로드된 파일: {file.filename}"
        )
    # 제한 파일용량 확인
    max_file_size = 260 * 1024  # 260KB
    if file.size > max_file_size:
        raise HTTPException(
            status_code=413,
            detail="파일 크기가 260KB보다 큽니다."
        )

    # UTF-8 형식인지 확인
    raw = await file.read()  # 1. 바이트로 읽기, 파일 관련 호출 앞에는 await 붙임
    try:
        text = raw.decode("utf-8")  # 2. 디코딩 시도
    except UnicodeDecodeError:  # 3. 실패하면 여기로 옴
        raise HTTPException(
            status_code=400,
            detail="UTF-8 형식만 지원합니다. 파일 인코딩을 UTF-8로 변환한 뒤 다시 업로드해주세요."
        )
    # 내용이 비어있진 않은지 확인
    if text.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="파일 내용이 비어있습니다."
        )

    # 실제 파이프라인 호출
    logger.info(f"[analysis] 분석 요청 — instructor={instructor_id} date={lecture_date}")

    try:
        scorecard = await analyze_raw_text(text, lecture_date, instructor_id)
    except Exception as e:
        logger.error(f"[analysis] 파이프라인 실패 — instructor={instructor_id}")
        raise HTTPException(
            status_code=502,
            detail=f"분석 파이프라인 실행 중 오류가 발생했습니다: {type(e).__name__}"
        )
    logger.info(f"[analysis] 분석 완료 — instructor={instructor_id} overall={scorecard.overall_score}")

    return scorecard
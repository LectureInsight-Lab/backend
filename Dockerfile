# LectureInsight Backend — FastAPI 분석 API
#
# 빌드 (EC2 x86_64 대상 — Apple Silicon 맥에서는 --platform 필수):
#   docker build --platform linux/amd64 -t lectureinsight-api .
#
# 실행:
#   docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data lectureinsight-api

FROM python:3.11-slim

# ── 환경변수 ───────────────────────────────────────────────
# PYTHONDONTWRITEBYTECODE: .pyc 파일 생성 안 함 (이미지 용량 절약)
# PYTHONUNBUFFERED:        stdout 버퍼링 끔 → docker logs에 로그가 즉시 보임 (중요!)
# TZ: 컨테이너 기본은 UTC라 analyzed_at·스코어카드 저장 날짜가 9시간 어긋난다.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Seoul

WORKDIR /app

# ── 시스템 패키지 ──────────────────────────────────────────
# kiwipiepy 등 C 확장 빌드용. wheel이 있으면 안 쓰이지만 없을 때를 대비.
# 같은 RUN에서 apt 캐시까지 지워야 레이어에 캐시가 안 남는다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── torch CPU 버전 먼저 설치 (핵심 최적화) ──────────────────
# sentence-transformers가 torch를 의존성으로 끌어오는데,
# 기본 PyPI의 torch는 CUDA 포함 버전이라 ~2.5GB.
# EC2에 GPU가 없으므로 CPU 전용 인덱스에서 먼저 설치해 ~200MB로 줄인다.
# (먼저 설치해두면 이후 pip이 이미 만족된 것으로 보고 CUDA 버전을 안 받는다)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ── 의존성 설치 (레이어 캐싱) ───────────────────────────────
# 코드보다 pyproject.toml을 먼저 복사하는 이유:
# 코드만 바뀌면 이 레이어는 캐시가 재사용돼 재빌드가 몇 초로 끝난다.
# (순서를 반대로 하면 코드 한 줄 고칠 때마다 전체 재설치 = 매번 15분)
COPY pyproject.toml README.md ./
RUN mkdir -p app && touch app/__init__.py \
    && pip install --no-cache-dir ".[embeddings]"

# ── 애플리케이션 코드 ───────────────────────────────────────
COPY app/ ./app/
COPY configs/ ./configs/

# ── 데이터 디렉토리 ────────────────────────────────────────
# store.py의 PROCESSED_DIR이 상대경로("data/processed")라 WORKDIR 기준으로 생성.
# 실제 운영에선 -v 로 호스트 볼륨을 마운트해 결과를 영속화한다.
RUN mkdir -p data/raw data/processed

# ── 비루트 유저 ────────────────────────────────────────────
# root로 돌리면 컨테이너 탈출 시 피해가 커진다. 보안 기본.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# ── 헬스체크 ───────────────────────────────────────────────
# 컨테이너가 "떠 있는지"가 아니라 "응답하는지"를 확인.
# 모델 로딩 때문에 첫 기동이 느릴 수 있어 start-period를 넉넉히.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --reload 없음: 프로덕션에서는 코드 감시가 불필요하고 메모리만 먹는다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

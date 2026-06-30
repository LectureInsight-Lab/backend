"""환경 설정 (Pydantic Settings).

`.env` 파일에서 LLM, RAG, 앙상블, 서버 설정을 로드한다.
v2: RAG(키워드 검색) + BoW 앙상블 시스템 설정.
LLM 제공자는 Gemini (google-genai). `.env` 의 ``API_KEY`` / ``LLM_MODEL`` 사용.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Server ─────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:8501"

    # ── LLM (Gemini) ───────────────────────────────────────
    api_key: str = ""                          # .env: API_KEY (Gemini API key)
    llm_model: str = "models/gemini-2.5-flash"  # .env: LLM_MODEL (production 채점)
    eval_model: str = "models/gemini-2.5-pro"   # .env: EVAL_MODEL (골든셋 라벨러/평가 — 강한 모델)
    llm_temperature: float = 0.2               # Paper #6: 일관된 채점
    # ── 호출 throttle (무료 티어 rate limit / 비용 절약) ─────
    llm_concurrency: int = 3        # 라벨링·항목 내부 동시 호출 수. 무료 티어면 1 권장
    llm_delay_sec: float = 0.0      # 호출 사이 지연(초). 무료 티어 RPM 회피용(예: 4~6)
    llm_thinking_budget: int = 0    # gemini-2.5-flash thinking 토큰 예산. 0=끔(비용↓), -1=자동

    # ── RAG (키워드 검색, 임베딩 없음) ──────────────────────
    rag_chunk_lines: int = 15             # Paper #1: 청크 크기(행)
    rag_top_k: int = 5                    # Paper #2: top-K 청크

    # ── 앙상블 (Paper #4) ───────────────────────────────────
    ensemble_llm_weight: float = 0.70
    ensemble_bow_weight: float = 0.30
    confidence_threshold: float = 0.5     # Paper #7

    # ── Cache ──────────────────────────────────────────────
    llm_cache_enabled: bool = True
    llm_cache_path: str = ".cache/llm"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

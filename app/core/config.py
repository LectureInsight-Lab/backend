"""환경 설정 (Pydantic Settings).

`.env` 파일에서 LLM, 임베딩, RAG, 앙상블, 서버 설정을 로드한다.
v2: RAG + BoW 앙상블 시스템에 필요한 설정 추가.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Server ─────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:8501"

    # ── LLM ────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2          # Paper #6: 일관된 채점

    # ── Embedding / RAG ────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    rag_chunk_lines: int = 15             # Paper #1
    rag_top_k: int = 5                    # Paper #2

    # ── 앙상블 (Paper #4) ───────────────────────────────────
    ensemble_llm_weight: float = 0.70
    ensemble_bow_weight: float = 0.30
    confidence_threshold: float = 0.5     # Paper #7

    # ── Cache ──────────────────────────────────────────────
    llm_cache_enabled: bool = True
    llm_cache_path: str = ".cache/llm"
    embedding_cache_path: str = "data/processed/embeddings"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

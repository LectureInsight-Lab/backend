from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"

    cors_origins: str = "http://localhost:3000"

    llm_cache_enabled: bool = True
    llm_cache_path: str = ".cache/llm"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

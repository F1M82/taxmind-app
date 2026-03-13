from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("config/.env")

class Settings(BaseSettings):
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    extraction_model: str = Field("llama-3.3-70b-versatile", env="EXTRACTION_MODEL")
    synthesis_model: str = Field("claude-sonnet-4-6", env="SYNTHESIS_MODEL")
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    qdrant_collection: str = Field("taxmind-judgments", env="QDRANT_COLLECTION")
    request_delay_seconds: float = Field(3.0, env="REQUEST_DELAY_SECONDS")
    max_requests_per_hour: int = Field(200, env="MAX_REQUESTS_PER_HOUR")
    respect_robots_txt: bool = Field(True, env="RESPECT_ROBOTS_TXT")
    user_agent: str = Field("TaxMind Research Bot (contact: gaurav@taxmind.in)", env="USER_AGENT")
    raw_data_dir: Path = Field(Path("data/raw"), env="RAW_DATA_DIR")
    extracted_data_dir: Path = Field(Path("data/extracted"), env="EXTRACTED_DATA_DIR")
    logs_dir: Path = Field(Path("data/logs"), env="LOGS_DIR")
    extraction_batch_size: int = Field(10, env="EXTRACTION_BATCH_SIZE")

    class Config:
        env_file = "config/.env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
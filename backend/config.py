from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    vllm_timeout_seconds: float = Field(default=120, gt=0)
    qasper_corpus_dir: Path = ROOT / "corpus" / "qasper"

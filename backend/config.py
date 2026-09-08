from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    glm_api_key: SecretStr = SecretStr("")
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-5.3"
    glm_timeout_seconds: float = Field(default=300, gt=0)
    glm_max_tokens: int = Field(default=16384, ge=1024)
    corpus_dir: Path = ROOT / "corpus"

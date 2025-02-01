from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Глобальный конфиг."""
    model_id: str
    hf_token: str

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings(_env_file="../.env", _env_nested_delimiter="__")

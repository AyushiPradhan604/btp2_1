import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    huggingfacehub_api_token: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "dummy")
    llm_model: str = os.getenv("LLM_MODEL", "google/gemma-2-9b-it")
    vision_model: str = os.getenv("VISION_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

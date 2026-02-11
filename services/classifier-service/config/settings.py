"""
Configuration management using Pydantic Settings.

Supports multiple LLM providers: Gemini, OpenAI, Anthropic, etc.
Switch providers by changing environment variables only.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    """
    Application settings with validation.
    
    All settings can be overridden via environment variables.
    """
    
    # LLM Provider Configuration
    llm_provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    
    # API Keys (only the one matching llm_provider is required)
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    
    # Model Configuration
    model_name: str = "gemini-2.5-flash"
    max_tokens: int = 1000
    temperature: float = 0.0  # Deterministic for classification
    
    # Service Configuration
    service_name: str = "classifier-service"
    service_version: str = "1.0.0"
    log_level: str = "INFO"
    
    # Classification Thresholds
    high_confidence_threshold: float = 0.75
    low_confidence_threshold: float = 0.50
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    def get_api_key(self) -> str:
        """
        Get the API key for the configured provider.
        
        Raises:
            ValueError: If API key for selected provider is not set
        """
        key_map = {
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key
        }
        
        api_key = key_map.get(self.llm_provider)
        
        if not api_key:
            raise ValueError(
                f"API key for provider '{self.llm_provider}' not found. "
                f"Please set {self.llm_provider.upper()}_API_KEY in your .env file"
            )
        
        return api_key


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    
    Using @lru_cache ensures we only load settings once,
    improving performance and ensuring consistency.
    """
    return Settings()
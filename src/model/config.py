"""
Configuration settings for model integration.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# load environment variables
load_dotenv()

class ModelConfig(BaseModel):
    """Configuration for the Claude model."""
    
    api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("MODEL_NAME", "claude-3-5-sonnet")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7"))
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024"))
    )
    top_p: Optional[float] = Field(
        default_factory=lambda: float(os.getenv("TOP_P", "1.0")) if os.getenv("TOP_P") else None
    )
    
    @field_validator('api_key')
    def validate_api_key(cls, v):
        """Validate that API key is provided."""
        if not v:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment variables")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

class LangGraphConfig(BaseModel):
    """Configuration for LangGraph."""
    
    checkpoint_dir: str = Field(
        default_factory=lambda: os.getenv("LANGGRAPH_CHECKPOINT_DIR", "./.langgraph")
    )
    tracing_enabled: bool = Field(
        default_factory=lambda: os.getenv("LANGGRAPH_TRACING_ENABLED", "false").lower() == "true"
    )
    project_name: str = Field(
        default_factory=lambda: os.getenv("LANGGRAPH_PROJECT_NAME", "memory_agent")
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

def get_model_config() -> ModelConfig:
    """
    Get the model configuration from environment variables.
    
    Returns:
        ModelConfig: The model configuration.
    """
    return ModelConfig()

def get_langgraph_config() -> LangGraphConfig:
    """
    Get the LangGraph configuration from environment variables.
    
    Returns:
        LangGraphConfig: The LangGraph configuration.
    """
    return LangGraphConfig()
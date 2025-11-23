from typing import Dict
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    model: str
    provider: str


class TemplateConfig(BaseModel):
    """Prompt template configuration."""
    prompt: str
    column_mapping: Dict[str, str]


class DatasetConfig(BaseModel):
    """Dataset input/output configuration."""
    input_file: str
    output_file: str


class ProcessingConfig(BaseModel):
    """Processing parameters configuration."""
    batch_size: int = Field(gt=0, description="Number of concurrent requests")
    retry: int = Field(ge=0, description="Number of retries for failed judgments")


class JudgeConfig(BaseModel):
    """Configuration for batch judging tasks."""
    llm: LLMConfig
    template: TemplateConfig
    dataset: DatasetConfig
    processing: ProcessingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'JudgeConfig':
        """Load and validate configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

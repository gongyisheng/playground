from typing import Dict, Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from omegaconf import OmegaConf


class MonteCarloConfig(BaseModel):
    """Monte Carlo judge method configuration."""
    num_rounds: int = Field(default=10, gt=0, description="Number of sampling rounds")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: int = Field(default=100, gt=0, description="Maximum tokens to generate")
    score_pattern: str = Field(default=r'^\d+$', description="Regex pattern to extract score")


class LogprobWeightedConfig(BaseModel):
    """Logprob weighted judge method configuration."""
    temperature: float = Field(default=0.0, ge=0, le=2, description="Sampling temperature")
    max_tokens: int = Field(default=1, gt=0, description="Maximum tokens to generate")


class JudgeMethodConfig(BaseModel):
    """Judge configuration combining LLM settings and judge parameters."""
    # LLM settings
    model: str
    provider: str

    # Judge prompt template
    prompt: str

    # Judge method and scoring
    method: Literal["logprob_weighted", "monte_carlo"] = "logprob_weighted"
    min_score: int = Field(default=0, description="Minimum score value")
    max_score: int = Field(default=100, description="Maximum score value")

    # Processing settings
    batch_size: int = Field(default=32, gt=0, description="Number of concurrent requests")
    retry: int = Field(default=3, ge=0, description="Number of retries for failed judgments")

    # Method-specific configs
    logprob_weighted: Optional[LogprobWeightedConfig] = None
    monte_carlo: Optional[MonteCarloConfig] = None


class DatasetConfig(BaseModel):
    """Dataset input/output configuration."""
    input_file: str
    output_file: str
    column_mapping: Dict[str, str] = Field(description="Map template variables to dataset columns")


class Config(BaseModel):
    """Configuration for batch judging tasks."""
    judge: JudgeMethodConfig
    dataset: DatasetConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load and validate configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load with OmegaConf to support variable interpolation
        config_omega = OmegaConf.load(path)
        # Convert to dict, resolving all interpolations
        config_dict = OmegaConf.to_container(config_omega, resolve=True)

        return cls(**config_dict)

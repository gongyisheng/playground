from typing import Dict, List, Any
from pathlib import Path
import yaml


class JudgeConfig:
    """Configuration for batch judging tasks."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._validate()

    def _validate(self):
        """Validate required fields in config."""
        required_fields = {
            'llm': ['model', 'provider'],
            'template': ['prompt', 'column_mapping'],
            'dataset': ['input_file', 'output_file'],
            'processing': ['batch_size', 'retry']
        }

        for section, fields in required_fields.items():
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")

            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(f"Missing required field: {section}.{field}")

        # Validate column_mapping is a dict
        if not isinstance(self.config['template']['column_mapping'], dict):
            raise ValueError("template.column_mapping must be a dictionary")

        # Validate batch_size and retry are positive integers
        if self.config['processing']['batch_size'] <= 0:
            raise ValueError("processing.batch_size must be positive")

        if self.config['processing']['retry'] < 0:
            raise ValueError("processing.retry must be non-negative")

    @property
    def model(self) -> str:
        return self.config['llm']['model']

    @property
    def provider(self) -> str:
        return self.config['llm']['provider']

    @property
    def prompt_template(self) -> str:
        return self.config['template']['prompt']

    @property
    def column_mapping(self) -> Dict[str, str]:
        return self.config['template']['column_mapping']

    @property
    def input_file(self) -> str:
        return self.config['dataset']['input_file']

    @property
    def output_file(self) -> str:
        return self.config['dataset']['output_file']

    @property
    def batch_size(self) -> int:
        return self.config['processing']['batch_size']

    @property
    def retry(self) -> int:
        return self.config['processing']['retry']

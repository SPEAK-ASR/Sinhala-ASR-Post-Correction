from dataclasses import dataclass

from config.model import ModelConfig
from config.training import TrainingConfig
from config.data import DataConfig


@dataclass
class Config:
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None

    def __post_init__(self) -> None:
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()


CONFIG = Config()

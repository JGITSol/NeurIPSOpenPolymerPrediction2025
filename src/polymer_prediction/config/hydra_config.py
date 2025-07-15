"""Hydra configuration for polymer prediction."""

from dataclasses import dataclass
from typing import Optional, List, Any
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gcn"
    hidden_channels: int = 128
    num_gcn_layers: int = 3
    dropout: float = 0.1
    activation: str = "relu"
    batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: Optional[str] = None
    target_column: str = "target_property"
    smiles_column: str = "smiles"
    test_split: float = 0.2
    val_split: float = 0.1
    random_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class FeaturizationConfig:
    """Featurization configuration."""
    add_hydrogens: bool = True
    use_chirality: bool = True
    use_bond_features: bool = True
    max_atoms: int = 200
    atom_features: List[str] = None
    
    def __post_init__(self):
        if self.atom_features is None:
            self.atom_features = [
                "atomic_num",
                "degree",
                "total_num_hs",
                "implicit_valence",
                "is_aromatic",
                "chiral_tag",
            ]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs"
    log_file: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "polymer-prediction"
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    name: str = "polymer_prediction"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    output_dir: str = "outputs"
    
    # Sub-configurations
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    featurization: FeaturizationConfig = FeaturizationConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Hydra specific
    hydra: Any = MISSING
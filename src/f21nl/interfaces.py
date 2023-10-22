from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple

import torch


class ModelInstance(NamedTuple):
    """A tuple provided to the model for each given step.

    Regardless of the step being performed, the model should always receive something for the input
    tokens. However, as the target tokens are not available during testing/predicting/inference,
    they might not be provided.
    """

    source_tokens: torch.Tensor
    source_attention_mask: torch.Tensor
    target_tokens: torch.Tensor | None = None
    target_attention_mask: torch.Tensor | None = None


CellType = Literal["gru", "lstm"]


class DatasetSize(str, Enum):
    """Identifiers for the different dataset sizes supported in this codebase."""

    large = "multi30k-en-fr-large"
    small = "multi30k-en-fr-small"


class EncoderConfig(NamedTuple):
    """Encoder config."""

    encoder_cell_type: CellType = "gru"
    encoder_input_dim: int = 50
    encoder_output_dim: int = 200
    source_embedding_dim: int = 50
    bidirectional: bool = False


class DecoderConfig(NamedTuple):
    """Decoder config."""

    decoder_cell_type: CellType = "gru"
    decoder_input_dim: int = 50
    decoder_output_dim: int = 200
    target_embedding_dim: int = 50


class DecodingStrategyConfig(NamedTuple):
    """Decoding Strategy config."""

    name: Literal["greedy", "top_k", "top_p"] = "greedy"
    top_k: int = 10
    top_p: float = 0.95


class EncoderDecoderConfig(NamedTuple):
    """Defines a set of predefined configs for your model."""

    encoder_config: EncoderConfig = EncoderConfig()
    decoder_config: DecoderConfig = DecoderConfig()
    use_bleu: bool = False
    attention: Literal["dot", "bilinear"] | None = None
    decoding_strategy: DecodingStrategyConfig = DecodingStrategyConfig()


class TrainConfig(NamedTuple):
    """Training configuration."""

    train_epochs: int = 10
    checkpoint_dirpath: Path = Path("storage/checkpoints/")
    use_early_stopping: bool = False
    early_stopping_monitor: str = "valid_loss"
    eary_stopping_mode: str = "min"


class DataConfig(NamedTuple):
    """Data configration."""

    dataset_size: DatasetSize = DatasetSize.small
    storage_dir: Path = Path("storage/data")
    batch_size: int = 4
    num_workers: int = 0

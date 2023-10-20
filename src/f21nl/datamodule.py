from pathlib import Path
from typing import Literal

import torch
from lightning import LightningDataModule
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from f21nl.data_preparation import run_data_preparation
from f21nl.dataset import Multi30kDataset
from f21nl.interfaces import DatasetSize, ModelInstance


SetupStage = Literal["fit", "validate", "test", "predict"]


class CollateFn:
    """Collate function used to generate tensors for the batch."""

    def __call__(self, instances: list[ModelInstance]) -> ModelInstance:
        """Uses pad_sequence to generate batched tensors."""
        source_tokens = torch.nn.utils.rnn.pad_sequence(
            [instance.source_tokens for instance in instances], batch_first=True, padding_value=0
        )

        source_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [instance.source_attention_mask for instance in instances],
            batch_first=True,
            padding_value=0,
        )

        raw_target_tokens = [
            instance.target_tokens for instance in instances if instance.target_tokens is not None
        ]

        if raw_target_tokens:
            target_tokens = torch.nn.utils.rnn.pad_sequence(
                raw_target_tokens, batch_first=True, padding_value=0
            )
        else:
            target_tokens = None

        raw_target_mask = [
            instance.target_attention_mask
            for instance in instances
            if instance.target_attention_mask is not None
        ]

        if raw_target_mask:
            target_attention_mask = torch.nn.utils.rnn.pad_sequence(
                raw_target_mask,
                batch_first=True,
                padding_value=0,
            )
        else:
            target_attention_mask = None

        return ModelInstance(
            source_tokens=source_tokens,
            source_attention_mask=source_attention_mask,
            target_tokens=target_tokens,
            target_attention_mask=target_attention_mask,
        )


class Multi30kDataModule(LightningDataModule):
    """A datamodule for the Multi30k dataset."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        storage_dir: Path,
        dataset_size: DatasetSize = DatasetSize.small,
        force_data_preparation: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        drop_last: bool = True,
        language_pairs: dict[str, str] | None = None,
    ) -> None:
        if language_pairs is None:
            language_pairs = {"source": "en", "target": "fr"}
        super().__init__()
        self._storage_dir = storage_dir
        self._force_data_preparation = force_data_preparation
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._drop_last = drop_last
        self._dataset_size = dataset_size
        self.train_dataset: Dataset[ModelInstance]
        self.valid_dataset: Dataset[ModelInstance]
        self.source_tokenizer: Tokenizer
        self.target_tokenizer: Tokenizer
        self._language_pairs = language_pairs

    def prepare_data(self) -> None:
        """Prepare the data on a single node.

        Do not assign anything to `self` within this method.
        """
        run_data_preparation(self._storage_dir, overwrite_existing=self._force_data_preparation)

    def setup(self, stage: SetupStage) -> None:
        """Setup the datamodule with the datasets."""
        if stage not in {"fit", "test", "predict"}:
            raise NotImplementedError(
                f"DataModule does not currently support the given stage `{stage}`"
            )

        source_tokenizer_file = self._storage_dir.joinpath(DatasetSize.large.value).joinpath(
            f"tokenizer-{self._language_pairs['source']}"
        )
        self.source_tokenizer = Tokenizer.from_file(source_tokenizer_file.as_posix())
        target_tokenizer_file = self._storage_dir.joinpath(DatasetSize.large.value).joinpath(
            f"tokenizer-{self._language_pairs['target']}"
        )
        self.target_tokenizer = Tokenizer.from_file(target_tokenizer_file.as_posix())

        if stage == "fit":
            source_data_file = self._storage_dir.joinpath(self._dataset_size.value).joinpath(
                f"train.lc.norm.tok.{self._language_pairs['source']}"
            )

            with Path.open(source_data_file) as in_file:
                source_data = in_file.readlines()

            target_data_file = self._storage_dir.joinpath(self._dataset_size.value).joinpath(
                f"train.lc.norm.tok.{self._language_pairs['target']}"
            )

            with Path.open(target_data_file) as in_file:
                target_data = in_file.readlines()

            self.train_dataset = Multi30kDataset(
                source_data=source_data,
                target_data=target_data,
                source_tokenizer=self.source_tokenizer,
                target_tokenizer=self.target_tokenizer,
            )

            source_data_file = self._storage_dir.joinpath(self._dataset_size.value).joinpath(
                f"val.lc.norm.tok.{self._language_pairs['source']}"
            )

            with Path.open(source_data_file) as in_file:
                source_data = in_file.readlines()

            target_data_file = self._storage_dir.joinpath(self._dataset_size.value).joinpath(
                f"val.lc.norm.tok.{self._language_pairs['target']}"
            )

            with Path.open(target_data_file) as in_file:
                target_data = in_file.readlines()

            self.valid_dataset = Multi30kDataset(
                source_data=source_data,
                target_data=target_data,
                source_tokenizer=self.source_tokenizer,
                target_tokenizer=self.target_tokenizer,
            )

        if stage == "test":
            source_data_file = self._storage_dir.joinpath(self._dataset_size.value).joinpath(
                f"val.lc.norm.tok.{self._language_pairs['source']}"
            )

            with Path.open(source_data_file) as in_file:
                source_data = in_file.readlines()

            target_data_file = self._storage_dir.joinpath(self._dataset_size.value).joinpath(
                f"val.lc.norm.tok.{self._language_pairs['target']}"
            )

            with Path.open(target_data_file) as in_file:
                target_data = in_file.readlines()

            self.test_dataset = Multi30kDataset(
                source_data=source_data,
                target_data=target_data,
                source_tokenizer=self.source_tokenizer,
                target_tokenizer=self.target_tokenizer,
            )

    def train_dataloader(self) -> DataLoader[ModelInstance]:
        """Return a dataloader over the training instances."""
        return DataLoader[ModelInstance](
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
            collate_fn=CollateFn(),
        )

    def val_dataloader(self) -> DataLoader[ModelInstance]:
        """Return a dataloader over the validation instances."""
        return DataLoader[ModelInstance](
            self.valid_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=False,
            collate_fn=CollateFn(),
        )

    def test_dataloader(self) -> DataLoader[ModelInstance]:
        """Return a dataloader over the validation instances."""
        return DataLoader[ModelInstance](
            self.test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=False,
            collate_fn=CollateFn(),
        )

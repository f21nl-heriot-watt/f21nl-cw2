import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from f21nl.interfaces import ModelInstance


class Multi30kDataset(Dataset[ModelInstance]):
    """Dataset class to read and preprocess data from the Multi30K dataset."""

    def __init__(
        self,
        source_data: list[str],
        target_data: list[str],
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
    ) -> None:
        super().__init__()

        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer
        self._source_data = source_data
        self._target_data = target_data

    def __len__(self) -> int:
        """Returns the number of data points in the dataset."""
        return len(self._source_data)

    def __getitem__(self, index: int) -> ModelInstance:
        """Returns a model instance combining both source and target tokens."""
        raw_source = self._source_data[index]
        raw_target = self._target_data[index]

        tokenized_source = self._source_tokenizer.encode(raw_source)
        tokenized_target = self._target_tokenizer.encode(raw_target)

        return ModelInstance(
            source_tokens=torch.tensor(tokenized_source.ids, dtype=torch.long),
            source_attention_mask=torch.tensor(tokenized_source.attention_mask, dtype=torch.bool),
            target_tokens=torch.tensor(tokenized_target.ids, dtype=torch.long),
            target_attention_mask=torch.tensor(tokenized_target.attention_mask, dtype=torch.bool),
        )

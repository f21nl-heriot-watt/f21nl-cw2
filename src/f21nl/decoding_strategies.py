from typing import Any

import torch
import torch.nn.functional as F

from f21nl.interfaces import EncoderDecoderConfig


class GreedyDecoding:
    """Greedy decoding strategy."""

    def __call__(self, step_logits: torch.Tensor) -> torch.Tensor:
        """Simply pick the word with the highest probability."""
        # shape: (batch_size, num_classes)
        class_probabilities = F.softmax(step_logits, dim=-1)

        # shape (predicted_classes): (batch_size,)
        _, predicted_classes = torch.max(class_probabilities, 1)

        return predicted_classes


class TopKDecoding:
    """TopK decoding strategy."""

    def __init__(self, top_k: int = 2, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._top_k = top_k
        self._filter_value = -float("Inf")

    def __call__(self, step_logits: torch.Tensor) -> torch.Tensor:
        """TODO: Implement this method.

        Order the words in descending order of probability;
        select the first K words to create a new distribution;
        sample from those tokens (Tip: use torch.multinomial())
        """
        raise NotImplementedError("Implement top-k decoding!")


class TopPDecoding:
    """TopP decoding strategy."""

    def __init__(self, top_p: float = 0.95, **kwargs: dict[str, Any]) -> torch.Tensor:
        if top_p < 0.0 or top_p > 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0!")

        self._top_p = top_p
        self._filter_value = -float("Inf")

    def __call__(self, step_logits: torch.Tensor) -> torch.Tensor:
        """TODO: Implement this method.

        Order the words in descending order of probability;
        Select the smallest number of top tokens such that their cumulative probability is at least p;
        Sample from those tokens as before.
        """
        raise NotImplementedError("Implement top-p decoding!")


class DecodingStrategy:
    """Decoding strategy."""

    def __init__(self, config: EncoderDecoderConfig) -> None:
        if config.decoding_strategy.name == "greedy":
            self._strategy = GreedyDecoding()
        elif config.decoding_strategy.name == "top_k":
            self._strategy = TopKDecoding(top_k=config.decoding_strategy.top_k)
        elif config.decoding_strategy.name == "top_p":
            self._strategy = TopPDecoding(top_p=config.decoding_strategy.top_p)
        else:
            raise ValueError(f"Decoding strategy {config.decoding_strategy.name} not supported!")

    def __call__(self, step_logits: torch.Tensor) -> torch.Tensor:
        return self._strategy(step_logits)

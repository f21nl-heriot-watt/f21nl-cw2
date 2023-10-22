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


class DecodingStrategy:
    """Decoding strategy."""

    def __init__(self, config: EncoderDecoderConfig) -> None:
        pass

    def __call__(self, step_logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "You must implement the __call__ method for the decoding strategy!"
        )

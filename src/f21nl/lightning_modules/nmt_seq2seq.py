from collections.abc import Callable, Iterator
from typing import Any, Optional

import numpy as np
import torch
from lightning import LightningModule
from tokenizers import Tokenizer
from torchmetrics.text import BLEUScore

from f21nl.interfaces import EncoderDecoderConfig, ModelInstance
from f21nl.model_components.encoder_decoder import NMTEncoderDecoder


OptimizerPartialFn = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]

_default_optimizer = torch.optim.Adam


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """Returns a moderately tiny value for a given PyTorch data type.

    This is used to avoid numerical issues such as division by zero. This is different from
    `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs. Only supports floating point
    dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def masked_mean(
    vector: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """To calculate mean along certain dimensions on masked values.

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


class NMTModule(LightningModule):
    """An example lightning module."""

    def __init__(
        self,
        config: EncoderDecoderConfig,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        *,
        optimizer_partial_fn: OptimizerPartialFn = _default_optimizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.model = None
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

        self._optimizer_partial_fn = optimizer_partial_fn

        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer

        if config.use_bleu:
            self._bleu = BLEUScore()
            self._predictions = []
            self._references = []
        else:
            self._bleu = None

        self.model = NMTEncoderDecoder(
            self.config,
            source_tokenizer=self._source_tokenizer,
            target_tokenizer=self._target_tokenizer,
        )

        self._end_index = self._target_tokenizer.encode("[EOS]").ids[0]
        self.save_hyperparameters(config._asdict())

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns:
        -------
        Dict[str, torch.Tensor]
        """
        # run the encoder on the source tokens
        state = self.model.encoder(source_tokens)
        state["source_mask"] = source_attention_mask

        # prepare decoder state
        state = self.model.init_decoder_state(state)

        # The `_forward_loop` decodes the input sequence and computes the loss during training
        # and validation.
        output_dict = self.model.forward_loop(state, target_tokens)

        return output_dict

    def _get_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        batch_size, max_seqlen, vocab_size = logits.shape

        losses = self.loss_function(logits.view(-1, vocab_size), relevant_targets.view(-1))

        return masked_mean(losses.view(batch_size, max_seqlen), relevant_mask, -1).mean()

    def _decode(self, output_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            predicted_tokens = self._target_tokenizer.decode(indices)
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def training_step(self, batch: ModelInstance, _: int) -> torch.Tensor:
        """Perform a training step over the batch.

        The training step must return the computed loss between the input and target tokens.
        """
        output_dict = self.forward(
            batch.source_tokens, batch.source_attention_mask, batch.target_tokens
        )

        loss = self._get_loss(
            output_dict["logits"], batch.target_tokens, batch.target_attention_mask
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: ModelInstance, _: int) -> torch.Tensor:
        """Perform a validation step over the batch.

        The validation step must return the computed loss between the input and target tokens. This
        will likely be similar to the `training_step`, but it also might not be.
        """
        output_dict = self.forward(
            batch.source_tokens, batch.source_attention_mask, batch.target_tokens
        )

        loss = self._get_loss(
            output_dict["logits"], batch.target_tokens, batch.target_attention_mask
        )

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if self._bleu is not None and batch.target_tokens is not None:
            predicted_text = self._target_tokenizer.decode_batch(
                output_dict["predictions"].tolist()
            )
            target_text = self._target_tokenizer.decode_batch(batch.target_tokens[:, 1:].tolist())
            self._predictions.extend(predicted_text)
            self._references.extend(target_text)

        return loss

    def predict_step(self, batch: ModelInstance, _: int) -> Any:
        """Generates a translation for an input sequence."""
        output_dict = self.forward(batch.source_tokens, batch.source_attention_mask)

        return self._decode(output_dict)

    def test_step(self, batch: ModelInstance, _: int) -> Any:
        """Generates a translation for an input sequence."""
        output_dict = self.forward(batch.source_tokens, batch.source_attention_mask)
        output_dict = self._decode(output_dict)

        if self._bleu is not None and batch.target_tokens is not None:
            predicted_text = output_dict["predicted_tokens"]
            target_text = self._target_tokenizer.decode_batch(batch.target_tokens[:, 1:].tolist())
            self._predictions.extend(predicted_text)
            self._references.extend(target_text)
        return output_dict

    def configure_optimizers(self) -> Any:
        """Configure the optimizer for training."""
        optimizer = self._optimizer_partial_fn(self.parameters())
        return {"optimizer": optimizer}

    def on_validation_epoch_end(self) -> None:
        """Compute score and reset metrics after each validation epoch."""
        if not self._bleu:
            return super().on_validation_epoch_end()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            world_size = torch.distributed.get_world_size()
            predictions = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(predictions, self._predictions)

            references = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(references, self._references)
            if torch.distributed.get_rank() == 0:
                predictions = list(
                    itertools.chain.from_iterable([out["predictions"] for out in output])  # type: ignore[index]
                )
                references = list(
                    itertools.chain.from_iterable([out["references"] for out in output])  # type: ignore[index]
                )
                bleu_score = self._bleu(predictions, references)
        else:
            bleu_score = self._bleu(self._predictions, [[ref] for ref in self._references])

        self.log("valid_bleu_score", bleu_score, on_epoch=True, prog_bar=True)
        self._predictions = []
        self._references = []
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        """Compute score and reset metrics after each test epoch."""
        if not self._bleu:
            return super().on_test_epoch_end()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            world_size = torch.distributed.get_world_size()
            predictions = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(predictions, self._predictions)

            references = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(references, self._references)
            if torch.distributed.get_rank() == 0:
                predictions = list(
                    itertools.chain.from_iterable([out["predictions"] for out in output])  # type: ignore[index]
                )
                references = list(
                    itertools.chain.from_iterable([out["references"] for out in output])  # type: ignore[index]
                )
                bleu_score = self._bleu(predictions, references)
        else:
            bleu_score = self._bleu(self._predictions, [[ref] for ref in self._references])

        self.log("test_bleu_score", bleu_score, on_epoch=True, prog_bar=True)
        self._predictions = []
        self._references = []
        return super().on_test_epoch_end()

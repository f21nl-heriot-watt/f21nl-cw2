from typing import Optional

import torch
from torch import nn
from torch.nn import Embedding, GRUCell, LSTMCell

from f21nl.interfaces import CellType


class NMTDecoder(nn.Module):
    """A simple sequence decoder using either GRU or LSTM."""

    def __init__(
        self,
        decoder_cell_type: CellType,
        decoder_input_dim: int,
        decoder_output_dim: int,
        target_vocab_size: int,
        target_embedding_dim: int,
    ) -> None:
        super().__init__()
        self._encoder_input_dim = decoder_input_dim
        self._encoder_output_dim = decoder_output_dim

        self.embeddings = Embedding(target_vocab_size, target_embedding_dim)
        if decoder_cell_type == "gru":
            self._decoder_cell = GRUCell(self._encoder_input_dim, self._encoder_output_dim)
        elif decoder_cell_type == "lstm":
            self._decoder_cell = LSTMCell(self._encoder_input_dim, self._encoder_output_dim)
        else:
            raise ValueError(f"Dialogue encoder of type {decoder_cell_type} not supported yet!")

    def forward(
        self,
        decoder_input_ids: torch.Tensor = None,
        decoder_embeds: torch.Tensor = None,
        decoder_hidden: torch.Tensor = None,
        decoder_context: torch.Tensor = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Performs the forward pass over the input sequence."""
        if decoder_input_ids is None and decoder_embeds is None:
            raise ValueError("Either decoder_input_ids or decoder_embeds must be provided.")

        if decoder_embeds is None:
            decoder_embeds = self.embeddings(decoder_input_ids)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        # Only LSTM has a decoder context; check for this
        if isinstance(self._decoder_cell, LSTMCell):
            hidden_state = (decoder_hidden, decoder_context)
            new_decoder_hidden, new_decoder_context = self._decoder_cell(
                decoder_embeds, hidden_state
            )

        else:
            hidden_state = decoder_hidden
            new_decoder_hidden = self._decoder_cell(decoder_embeds, hidden_state)
            new_decoder_context = None

        return new_decoder_hidden, new_decoder_context

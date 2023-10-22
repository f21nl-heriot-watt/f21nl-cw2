import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn
from torch.nn import Linear

from f21nl.decoding_strategies import GreedyDecoding
from f21nl.interfaces import EncoderDecoderConfig
from f21nl.model_components.decoder import NMTDecoder
from f21nl.model_components.encoder import NMTEncoder


class NMTEncoderDecoder(nn.Module):
    def __init__(
        self,
        config: EncoderDecoderConfig,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        max_decoding_steps: int = 20,
    ) -> None:
        super().__init__()

        self._config = config

        self.encoder: NMTEncoder = NMTEncoder(
            encoder_cell_type=config.encoder_config.encoder_cell_type,
            encoder_input_dim=config.encoder_config.encoder_input_dim,
            encoder_output_dim=config.encoder_config.encoder_output_dim,
            source_vocab_size=source_tokenizer.get_vocab_size(),
            source_embedding_dim=config.encoder_config.source_embedding_dim,
            bidirectional=config.encoder_config.bidirectional,
        )
        self.decoder: NMTDecoder = NMTDecoder(
            decoder_cell_type=config.decoder_config.decoder_cell_type,
            decoder_input_dim=config.decoder_config.decoder_input_dim,
            decoder_output_dim=config.decoder_config.decoder_output_dim,
            target_vocab_size=target_tokenizer.get_vocab_size(),
            target_embedding_dim=config.decoder_config.target_embedding_dim,
        )

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(
            config.decoder_config.decoder_output_dim, target_tokenizer.get_vocab_size()
        )

        self._start_index = target_tokenizer.encode("[BOS]").ids[0]
        self._max_decoding_steps = max_decoding_steps

        self._attention = config.attention

        if config.decoding_strategy.name == "greedy":
            self.decode_strategy = GreedyDecoding()
        else:
            raise NotImplementedError("You still haven't implemented your decoding strategy!")

    def take_step(
        self, last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns:
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes:
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state, _ = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    def init_decoder_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = self.encoder.get_final_encoder_states(
            state["encoder_outputs"], state["source_mask"]
        )
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self._config.decoder_config.decoder_output_dim
        )
        return state

    def forward_loop(
        self,
        state: dict[str, torch.Tensor],
        target_tokens: torch.LongTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Make forward pass during training or do greedy search during prediction.

        Notes:
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens is not None:
            _, target_sequence_length = target_tokens.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        step_logits: list[torch.Tensor] = []
        step_predictions: list[torch.Tensor] = []
        step_attentions: list[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size,)
            if target_tokens is not None:
                input_choices = target_tokens[:, timestep]
            else:
                input_choices = last_predictions

            # shape: (batch_size, num_classes)
            output_projections, state, attention_scores = self._prepare_output_projections(
                input_choices, state
            )

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            predicted_classes = self.decode_strategy(output_projections)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))
            if attention_scores is not None:
                step_attentions.append(attention_scores.unsqueeze(0))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        attentions = None
        if step_attentions:
            attentions = torch.cat(step_attentions, 0).transpose(0, 1)
        output_dict = {"predictions": predictions, "attentions": attentions}

        if target_tokens is not None:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # removes NaN values for masked positions
            logits.nan_to_num_(0.0)

            output_dict["logits"] = logits

        return output_dict

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """Decoder prediction step.

        Decode current state and last prediction to produce projections into the target space,
        which can then be used to get probabilities of each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)

        attention_scores = None
        decoder_input = None
        decoder_embeds = None

        if self._attention:
            decoder_embeds = self.decoder.embeddings(last_predictions)

            attended_input, attention_scores = self._compute_attention(
                decoder_hidden, encoder_outputs, source_mask
            )
            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_embeds = torch.cat((attended_input, decoder_embeds), -1)

        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = last_predictions

        decoder_hidden, decoder_context = self.decoder(
            decoder_input, decoder_embeds, decoder_hidden, decoder_context
        )

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state, attention_scores

    # TODO: Implement attention mechanisms here
    def _compute_attention(
        self,
        decoder_hidden_state: torch.Tensor = None,
        encoder_outputs: torch.Tensor = None,
        encoder_outputs_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state.

        Parameters
        ----------
        decoder_hidden_state : ``torch.LongTensor``
            A tensor of shape ``(batch_size, decoder_output_dim)``, which contains the current decoder hidden state to be used
            as the 'query' to the attention computation
            during the last time step.
        encoder_outputs : ``torch.LongTensor``
            A tensor of shape ``(batch_size, max_input_sequence_length, encoder_output_dim)``, which contains all the
            encoder hidden states of the source tokens, i.e., the 'keys' to the attention computation
        encoder_mask : ``torch.LongTensor``
            A tensor of shape (batch_size, max_input_sequence_length), which contains the mask of the encoded input.
            We want to avoid computing an attention score for positions of the source with zero-values (remember not all
            input sentences have the same length).

        Returns:
        -------
        torch.Tensor
            A tensor of shape (batch_size, encoder_output_dim) that contains the attended encoder outputs (aka context vector),
            i.e., we have ``applied`` the attention scores on the encoder hidden states.
        """
        # Ensure mask is also a FloatTensor. Or else the multiplication within attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        # encoder_outputs_mask = encoder_outputs_mask.float()
        # Main body of attention weights computation here
        if self._attention == "dot_product":
            raise NotImplementedError("Implement the dot-product attention here!")
        elif self._attention == "bilinear":
            raise NotImplementedError("Implement bilinear attention here!")
        else:
            raise ValueError(f"Attention mechanism of type {self._attention} not supported yet!")

        # shape: (batch_size, input_sequence_length)
        # TODO: given the unnormalised attention score, use the softmax to normalise them
        # use the mask to make sure that you apply the softmax ignoring padding positions
        # to ignore padding positions use a very small floating value
        # masked_attention_scores = ...

        # Apply the attention weights on the encoder outputs
        # shape: (batch_size, encoder_output_dim)
        # TODO: re-weight each encoder hidden state using the attention weights
        # attended_input = ...

        # TODO: return attended_input and masked_attention_scores
        return attended_input, masked_attention_scores

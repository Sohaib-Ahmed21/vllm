# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

from transformers import PretrainedConfig


class NemotronAsrStreamingEncoderConfig(PretrainedConfig):
    """Configuration for the Nemotron Fast-Conformer encoder."""

    model_type = "nemotron_asr_streaming_encoder"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        intermediate_size: int = 4096,
        hidden_act: str = "silu",
        attention_bias: bool = True,
        convolution_bias: bool = True,
        conv_kernel_size: int = 9,
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        num_mel_bins: int = 80,
        subsampling_conv_kernel_size: int = 3,
        subsampling_conv_stride: int = 2,
        dropout: float = 0.1,
        dropout_positions: float = 0.0,
        layerdrop: float = 0.1,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 5000,
        scale_input: bool = True,
        initializer_range: float = 0.02,
        sliding_window: int = 71,
        default_num_lookahead_tokens: int = 13,
        num_key_value_heads: int | None = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError(
                "Nemotron ASR requires num_key_value_heads to equal "
                "num_attention_heads."
            )
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.convolution_bias = convolution_bias
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.num_mel_bins = num_mel_bins
        self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride
        self.dropout = dropout
        self.dropout_positions = dropout_positions
        self.layerdrop = layerdrop
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.scale_input = scale_input
        self.initializer_range = initializer_range
        self.sliding_window = sliding_window
        self.default_num_lookahead_tokens = default_num_lookahead_tokens

        super().__init__(**kwargs)

    @property
    def subsampling_out_hidden_size(self) -> int:
        """Return the flattened feature size after 2-D subsampling."""
        total_pad = (self.subsampling_conv_kernel_size - 1) + (
            self.subsampling_conv_stride - 1
        )
        out_length = self.num_mel_bins
        for _ in range(int(math.log2(self.subsampling_factor))):
            out_length = (
                out_length + total_pad - self.subsampling_conv_kernel_size
            ) // self.subsampling_conv_stride + 1
        return self.subsampling_conv_channels * out_length


class Nemotron3_5AsrConfig(PretrainedConfig):
    """Configuration for NVIDIA Nemotron 3.5 ASR."""

    model_type = "nemotron3_5_asr"
    sub_configs = {"encoder_config": NemotronAsrStreamingEncoderConfig}

    def __init__(
        self,
        vocab_size: int = 13088,
        decoder_hidden_size: int = 640,
        num_decoder_layers: int = 2,
        hidden_act: str = "relu",
        max_symbols_per_step: int = 10,
        encoder_config: dict | PretrainedConfig | None = None,
        pad_token_id: int = 0,
        blank_token_id: int = 13087,
        is_encoder_decoder: bool = True,
        num_prompts: int = 128,
        prompt_intermediate_size: int = 2048,
        default_prompt_id: int = 101,
        architectures: list[str] | None = None,
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            encoder_config = NemotronAsrStreamingEncoderConfig(**encoder_config)
        elif encoder_config is None:
            encoder_config = NemotronAsrStreamingEncoderConfig()

        self.encoder_config = encoder_config
        self.vocab_size = vocab_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.hidden_act = hidden_act
        self.max_symbols_per_step = max_symbols_per_step
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.num_prompts = num_prompts
        self.prompt_intermediate_size = prompt_intermediate_size
        self.default_prompt_id = default_prompt_id

        # Keep the common vLLM model-config attributes available on the
        # top-level config. The encoder is the only attention stack.
        self.hidden_size = encoder_config.hidden_size
        self.num_hidden_layers = encoder_config.num_hidden_layers
        self.num_attention_heads = encoder_config.num_attention_heads
        self.num_key_value_heads = encoder_config.num_key_value_heads
        self.intermediate_size = encoder_config.intermediate_size
        self.architectures = architectures or ["Nemotron3_5AsrForRNNT"]

        super().__init__(
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            is_encoder_decoder=is_encoder_decoder,
            architectures=self.architectures,
            **kwargs,
        )


__all__ = [
    "NemotronAsrStreamingEncoderConfig",
    "Nemotron3_5AsrConfig",
]

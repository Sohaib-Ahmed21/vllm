# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.nemotron3_5_asr import (
    Nemotron3_5AsrAudioEncoder,
)
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.configs.nemotron3_5_asr import (
    Nemotron3_5AsrConfig,
    NemotronAsrStreamingEncoderConfig,
)


def _get_tiny_config() -> Nemotron3_5AsrConfig:
    encoder_config = NemotronAsrStreamingEncoderConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        attention_bias=False,
        convolution_bias=False,
        conv_kernel_size=3,
        subsampling_factor=8,
        subsampling_conv_channels=4,
        num_mel_bins=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=32,
        scale_input=False,
        sliding_window=9,
        default_num_lookahead_tokens=3,
    )
    return Nemotron3_5AsrConfig(
        encoder_config=encoder_config,
        decoder_hidden_size=8,
        num_prompts=8,
        prompt_intermediate_size=16,
        default_prompt_id=3,
    )


def test_nemotron_config_builds_nested_encoder_config() -> None:
    config = Nemotron3_5AsrConfig(
        encoder_config={
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_mel_bins": 8,
            "subsampling_conv_channels": 4,
        }
    )

    assert isinstance(config.encoder_config, NemotronAsrStreamingEncoderConfig)
    assert config.encoder_config.hidden_size == 16
    assert config.encoder_config.subsampling_out_hidden_size == 8
    assert config.is_encoder_decoder
    assert _CONFIG_REGISTRY["nemotron3_5_asr"] is Nemotron3_5AsrConfig


def test_nemotron_config_rejects_grouped_query_attention() -> None:
    with pytest.raises(ValueError, match="num_key_value_heads to equal"):
        NemotronAsrStreamingEncoderConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
        )


def test_nemotron_audio_encoder_uses_configured_activation() -> None:
    config = _get_tiny_config()
    config.encoder_config.hidden_act = "relu"

    model = Nemotron3_5AsrAudioEncoder(config)
    layer = model.encoder.layers[0]

    assert isinstance(layer.feed_forward1.activation, torch.nn.ReLU)
    assert isinstance(layer.conv.activation, torch.nn.ReLU)


def test_nemotron_audio_encoder_preserves_batch_and_valid_lengths() -> None:
    torch.manual_seed(0)
    model = Nemotron3_5AsrAudioEncoder(_get_tiny_config()).eval()
    input_features = torch.randn(2, 26, 8)
    attention_mask = torch.zeros(2, 26, dtype=torch.bool)
    attention_mask[0, :25] = True
    attention_mask[1, :17] = True

    with torch.inference_mode():
        output, output_mask = model(
            input_features,
            attention_mask,
            prompt_ids=torch.tensor([2, 3]),
        )
        unpadded_output, unpadded_mask = model(
            input_features[1:2, :17],
            torch.ones(1, 17, dtype=torch.bool),
            prompt_ids=torch.tensor([3]),
        )

    assert output.shape == (2, 5, 8)
    assert output_mask is not None
    assert output_mask.sum(-1).tolist() == [4, 3]
    assert torch.isfinite(output).all()
    assert unpadded_mask is not None
    assert unpadded_mask.sum().item() == 3
    torch.testing.assert_close(output[1, :3], unpadded_output[0, :3])


def test_nemotron_audio_encoder_parameter_names_match_checkpoint() -> None:
    parameter_names = dict(
        Nemotron3_5AsrAudioEncoder(_get_tiny_config()).named_parameters()
    )

    assert "encoder.subsampling.conv_in.weight" in parameter_names
    assert "encoder.layers.0.self_attn.relative_k_proj.weight" in parameter_names
    assert "prompt_projector.linear_1.weight" in parameter_names
    assert "encoder_projector.weight" in parameter_names

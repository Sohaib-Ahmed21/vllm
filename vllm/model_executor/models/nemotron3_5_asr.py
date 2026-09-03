# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline input processing and encoder for NVIDIA Nemotron 3.5 ASR."""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_fn
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams
from vllm.transformers_utils.configs.nemotron3_5_asr import (
    Nemotron3_5AsrConfig,
    NemotronAsrStreamingEncoderConfig,
)
from vllm.transformers_utils.processors.nemotron3_5_asr import (
    Nemotron3_5AsrProcessor,
    NemotronAsrStreamingFeatureExtractor,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict


def _get_subsampling_output_lengths(
    input_lengths: torch.Tensor,
    *,
    subsampling_factor: int,
    subsampling_conv_kernel_size: int,
    subsampling_conv_stride: int,
) -> torch.Tensor:
    num_layers = int(math.log2(subsampling_factor))
    all_paddings = (subsampling_conv_kernel_size - 1) + (subsampling_conv_stride - 1)
    add_pad = all_paddings - subsampling_conv_kernel_size

    output_lengths = input_lengths
    for _ in range(num_layers):
        output_lengths = (
            torch.div(
                output_lengths + add_pad,
                subsampling_conv_stride,
                rounding_mode="floor",
            )
            + 1
        )

    return output_lengths


def _get_max_subsampling_input_length(
    output_length: int,
    *,
    subsampling_factor: int,
    subsampling_conv_kernel_size: int,
    subsampling_conv_stride: int,
) -> int:
    num_layers = int(math.log2(subsampling_factor))
    all_paddings = (subsampling_conv_kernel_size - 1) + (subsampling_conv_stride - 1)
    add_pad = all_paddings - subsampling_conv_kernel_size

    input_length = output_length
    for _ in range(num_layers):
        input_length = subsampling_conv_stride * input_length - add_pad - 1

    return input_length


class Nemotron3_5AsrProcessingInfo(BaseProcessingInfo):
    """vLLM metadata and preprocessing information for Nemotron ASR."""

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_hf_config(self) -> Nemotron3_5AsrConfig:
        return self.ctx.get_hf_config(Nemotron3_5AsrConfig)

    def get_hf_processor(self, **kwargs: object) -> Nemotron3_5AsrProcessor:
        del kwargs
        if not hasattr(self, "_cached_hf_processor"):
            revision = self.ctx.model_config.revision
            processor_config = get_hf_file_to_dict(
                "processor_config.json",
                self.model_id,
                revision=revision,
            )
            if processor_config is None:
                raise ValueError(
                    f"processor_config.json was not found for {self.model_id}."
                )
            processor_config = dict(processor_config)
            feature_config = dict(processor_config.pop("feature_extractor"))
            feature_config.pop("feature_extractor_type", None)
            processor_config.pop("processor_class", None)
            feature_extractor = NemotronAsrStreamingFeatureExtractor(**feature_config)
            self._cached_hf_processor = Nemotron3_5AsrProcessor(
                feature_extractor,
                self.get_tokenizer(),
                **processor_config,
            )
        return self._cached_hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
        )

    def get_feature_extractor(
        self, **kwargs: object
    ) -> NemotronAsrStreamingFeatureExtractor:
        processor = self.get_hf_processor(**kwargs)
        return processor.feature_extractor

    @property
    def skip_prompt_length_check(self) -> bool:
        return True


class Nemotron3_5AsrDummyInputsBuilder(
    BaseDummyInputsBuilder[Nemotron3_5AsrProcessingInfo]
):
    """Build a bounded audio input for multimodal profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        encoder_config = self.info.get_hf_config().encoder_config
        num_audios = mm_counts.get("audio", 0)

        max_mel_frames = _get_max_subsampling_input_length(
            encoder_config.max_position_embeddings,
            subsampling_factor=encoder_config.subsampling_factor,
            subsampling_conv_kernel_size=(encoder_config.subsampling_conv_kernel_size),
            subsampling_conv_stride=encoder_config.subsampling_conv_stride,
        )
        audio_len = max_mel_frames * feature_extractor.hop_length - 1
        audio_overrides = mm_options.get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class Nemotron3_5AsrMultiModalProcessor(
    EncDecMultiModalProcessor[Nemotron3_5AsrProcessingInfo]
):
    """Convert vLLM audio items into Nemotron encoder inputs."""

    skip_decoder_start_token: bool = True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [self.info.get_hf_config().blank_token_id]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            audios = mm_data["audios"]
            mm_data = {"audio": audios}
            mm_kwargs = {
                **mm_kwargs,
                "sampling_rate": feature_extractor.sampling_rate,
            }

        tok_kwargs = {
            key: value
            for key, value in tok_kwargs.items()
            if key not in ("truncation", "max_length")
        }

        processed = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        labels = processed.pop("labels", None)
        if labels is not None:
            processed["input_ids"] = labels
        processed.pop("decoder_input_ids", None)

        lookahead = processed.get("num_lookahead_tokens")
        if lookahead is not None:
            lookahead = torch.as_tensor(lookahead, dtype=torch.long).reshape(-1)
            if lookahead.numel() != 1:
                raise ValueError(
                    "Expected num_lookahead_tokens to be a single value, "
                    f"but received {lookahead.numel()} values."
                )
            processed["num_lookahead_tokens"] = lookahead.squeeze(0)

        return processed

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_audios = hf_inputs["input_features"].shape[0]
        return {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "attention_mask": MultiModalFieldConfig.batched("audio"),
            "prompt_ids": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            "num_lookahead_tokens": MultiModalFieldConfig.shared(
                "audio", num_audios, keep_on_cpu=True
            ),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        attention_mask = out_mm_kwargs.get_data()["attention_mask"]
        assert isinstance(attention_mask, torch.Tensor)

        encoder_config = self.info.get_hf_config().encoder_config
        output_lengths = _get_subsampling_output_lengths(
            attention_mask.sum(-1),
            subsampling_factor=encoder_config.subsampling_factor,
            subsampling_conv_kernel_size=(encoder_config.subsampling_conv_kernel_size),
            subsampling_conv_stride=encoder_config.subsampling_conv_stride,
        ).tolist()

        def replacement(item_idx: int) -> list[int]:
            return [0] * output_lengths[item_idx]

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=replacement,
            )
        ]


class NemotronAsrStreamingCausalConv1d(nn.Conv1d):
    """Causal convolution used by each Fast-Conformer block."""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        effective_kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        input_ = F.pad(input_, (effective_kernel - self.stride[0], 0))
        return super().forward(input_)


class NemotronAsrStreamingCausalConv2d(nn.Conv2d):
    """Causal time convolution with NeMo-compatible frequency padding."""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        time_padding = (self.kernel_size[0] - 1, self.stride[0] - 1)
        frequency_padding = (self.kernel_size[1] - 1, self.stride[1] - 1)
        input_ = F.pad(input_, (*frequency_padding, *time_padding))
        return super().forward(input_)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        total_padding = (self.kernel_size[0] - 1) + (self.stride[0] - 1)
        return (
            torch.div(
                input_lengths + total_padding - self.kernel_size[0],
                self.stride[0],
                rounding_mode="floor",
            )
            + 1
        )


def _mask_subsampled_frames(
    hidden_states: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    if lengths is None:
        return hidden_states
    time = torch.arange(hidden_states.shape[2], device=hidden_states.device)
    mask = time < lengths[:, None]
    return hidden_states * mask[:, None, :, None]


class NemotronAsrStreamingSubsamplingLayer(nn.Module):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()
        channels = config.subsampling_conv_channels
        self.depthwise_conv = NemotronAsrStreamingCausalConv2d(
            channels,
            channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
            groups=channels,
        )
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = self.depthwise_conv(hidden_states)
        if lengths is not None:
            lengths = self.depthwise_conv.output_lengths(lengths)
        hidden_states = self.pointwise_conv(hidden_states)
        return _mask_subsampled_frames(hidden_states, lengths), lengths


class NemotronAsrStreamingSubsamplingConv2d(nn.Module):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()
        channels = config.subsampling_conv_channels
        num_layers = int(math.log2(config.subsampling_factor))
        self.conv_in = NemotronAsrStreamingCausalConv2d(
            1,
            channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.layers = nn.ModuleList(
            NemotronAsrStreamingSubsamplingLayer(config) for _ in range(1, num_layers)
        )
        self.act_fn = nn.ReLU()
        self.linear = nn.Linear(
            config.subsampling_out_hidden_size,
            config.hidden_size,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = input_features.unsqueeze(1)
        lengths = attention_mask.sum(-1) if attention_mask is not None else None

        hidden_states = self.conv_in(hidden_states)
        if lengths is not None:
            lengths = self.conv_in.output_lengths(lengths)
        hidden_states = self.act_fn(_mask_subsampled_frames(hidden_states, lengths))

        for layer in self.layers:
            hidden_states, lengths = layer(hidden_states, lengths)
            hidden_states = self.act_fn(hidden_states)

        batch_size, channels, time, frequency = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size,
            time,
            channels * frequency,
        )
        return self.linear(hidden_states)


class NemotronAsrStreamingRelativePositionEncoding(nn.Module):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, config.hidden_size, 2, dtype=torch.float32)
                / config.hidden_size
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[1]
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Encoder sequence length {seq_length} exceeds "
                f"max_position_embeddings={self.max_position_embeddings}."
            )

        positions = torch.arange(
            seq_length - 1,
            -seq_length,
            -1,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(positions, self.inv_freq.float())
        embeddings = torch.stack(
            (frequencies.sin(), frequencies.cos()),
            dim=-1,
        ).flatten(-2)
        return (
            embeddings[None]
            .expand(hidden_states.shape[0], -1, -1)
            .to(hidden_states.dtype)
        )


class NemotronAsrStreamingAttention(nn.Module):
    def __init__(
        self,
        config: NemotronAsrStreamingEncoderConfig,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.relative_k_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    @staticmethod
    def _relative_shift(attention_scores: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, query_length, position_length = attention_scores.shape
        attention_scores = F.pad(attention_scores, (1, 0))
        attention_scores = attention_scores.view(
            batch_size,
            num_heads,
            -1,
            query_length,
        )
        return attention_scores[:, :, 1:].view(
            batch_size,
            num_heads,
            query_length,
            position_length,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        projected_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(projected_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(projected_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(projected_shape).transpose(1, 2)

        query_content = query + self.bias_u[None, :, None, :]
        query_position = query + self.bias_v[None, :, None, :]
        relative_key = self.relative_k_proj(position_embeddings).view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim,
        )

        position_scores = query_position @ relative_key.permute(0, 2, 3, 1)
        position_scores = self._relative_shift(position_scores)[..., :seq_length]
        attention_scores = (query_content @ key.transpose(2, 3)) * self.scaling
        attention_scores += position_scores * self.scaling
        if attention_mask is not None:
            attention_scores.masked_fill_(~attention_mask, float("-inf"))

        probabilities = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        probabilities = probabilities.to(query.dtype)
        probabilities = F.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        )
        output = probabilities @ value
        output = output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        return self.o_proj(output)


class NemotronAsrStreamingFeedForward(nn.Module):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.linear2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.activation = get_act_fn(config.hidden_act)
        self.activation_dropout = config.activation_dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.linear1(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        return self.linear2(hidden_states)


class NemotronAsrStreamingConvolutionModule(nn.Module):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()
        channels = config.hidden_size
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            bias=config.convolution_bias,
        )
        self.depthwise_conv = NemotronAsrStreamingCausalConv1d(
            channels,
            channels,
            config.conv_kernel_size,
            groups=channels,
            bias=config.convolution_bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=config.convolution_bias,
        )
        self.activation = get_act_fn(config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = F.glu(
            self.pointwise_conv1(hidden_states.transpose(1, 2)),
            dim=1,
        )
        if padding_mask is not None:
            hidden_states.masked_fill_(padding_mask[:, None, :], 0.0)
        hidden_states = self.depthwise_conv(hidden_states).transpose(1, 2)
        hidden_states = self.norm(hidden_states).transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        return self.pointwise_conv2(hidden_states).transpose(1, 2)


class NemotronAsrStreamingEncoderBlock(nn.Module):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()
        self.feed_forward1 = NemotronAsrStreamingFeedForward(config)
        self.self_attn = NemotronAsrStreamingAttention(config)
        self.conv = NemotronAsrStreamingConvolutionModule(config)
        self.feed_forward2 = NemotronAsrStreamingFeedForward(config)
        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states
        hidden_states = hidden_states + self.self_attn(
            self.norm_self_att(hidden_states),
            position_embeddings,
            attention_mask,
        )
        hidden_states = hidden_states + self.conv(
            self.norm_conv(hidden_states),
            padding_mask,
        )
        hidden_states = hidden_states + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden_states)
        )
        return self.norm_out(hidden_states)


class NemotronAsrStreamingEncoder(nn.Module):
    """Offline Fast-Conformer encoder for Nemotron 3.5 ASR."""

    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()
        self.config = config
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = NemotronAsrStreamingSubsamplingConv2d(config)
        self.encode_positions = NemotronAsrStreamingRelativePositionEncoding(config)
        self.layers = nn.ModuleList(
            NemotronAsrStreamingEncoderBlock(config)
            for _ in range(config.num_hidden_layers)
        )

    def _get_output_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        output_lengths = _get_subsampling_output_lengths(
            attention_mask.sum(-1),
            subsampling_factor=self.config.subsampling_factor,
            subsampling_conv_kernel_size=(self.config.subsampling_conv_kernel_size),
            subsampling_conv_stride=self.config.subsampling_conv_stride,
        )
        positions = torch.arange(target_length, device=attention_mask.device)
        return positions < output_lengths[:, None]

    def _get_attention_mask(
        self,
        output_mask: torch.Tensor | None,
        seq_length: int,
        num_lookahead_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        chunk_size = num_lookahead_tokens + 1
        left_context_chunks = (self.config.sliding_window - 1) // chunk_size
        positions = torch.arange(seq_length, device=device)
        chunks = torch.div(positions, chunk_size, rounding_mode="floor")
        chunk_difference = chunks[:, None] - chunks[None, :]
        chunk_mask = (chunk_difference >= 0) & (chunk_difference <= left_context_chunks)
        attention_mask = chunk_mask[None, None]
        if output_mask is not None:
            attention_mask = attention_mask & output_mask[:, None, None, :]
        return attention_mask

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_lookahead_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states *= self.input_scale
        seq_length = hidden_states.shape[1]
        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(
                attention_mask,
                seq_length,
            )
        if num_lookahead_tokens is None:
            num_lookahead_tokens = self.config.default_num_lookahead_tokens
        chunk_mask = self._get_attention_mask(
            output_mask,
            seq_length,
            num_lookahead_tokens,
            hidden_states.device,
        )
        positions = self.encode_positions(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p=self.config.dropout,
            training=self.training,
        )
        positions = F.dropout(
            positions,
            p=self.config.dropout_positions,
            training=self.training,
        )
        padding_mask = ~output_mask if output_mask is not None else None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                chunk_mask,
                padding_mask,
                positions,
            )
        return hidden_states, output_mask


class Nemotron3_5AsrPromptProjector(nn.Module):
    def __init__(self, config: Nemotron3_5AsrConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.encoder_config.hidden_size + config.num_prompts,
            config.prompt_intermediate_size,
        )
        self.act = nn.ReLU()
        self.linear_2 = nn.Linear(
            config.prompt_intermediate_size,
            config.encoder_config.hidden_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(hidden_states)))


class Nemotron3_5AsrAudioEncoder(nn.Module):
    """Encode mel features and apply language-prompt conditioning."""

    def __init__(self, config: Nemotron3_5AsrConfig):
        super().__init__()
        self.config = config
        self.encoder = NemotronAsrStreamingEncoder(config.encoder_config)
        self.encoder_projector = nn.Linear(
            config.encoder_config.hidden_size,
            config.decoder_hidden_size,
        )
        self.prompt_projector = Nemotron3_5AsrPromptProjector(config)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_ids: torch.LongTensor | None = None,
        num_lookahead_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states, output_mask = self.encoder(
            input_features,
            attention_mask,
            num_lookahead_tokens,
        )
        if prompt_ids is None:
            prompt_ids = torch.full(
                (hidden_states.shape[0],),
                self.config.default_prompt_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
        prompt_ids = prompt_ids.to(hidden_states.device)
        prompt = F.one_hot(
            prompt_ids,
            num_classes=self.config.num_prompts,
        ).to(hidden_states.dtype)
        prompt = prompt[:, None, :].expand(-1, hidden_states.shape[1], -1)
        hidden_states = self.prompt_projector(
            torch.cat((hidden_states, prompt), dim=-1)
        )
        return self.encoder_projector(hidden_states), output_mask

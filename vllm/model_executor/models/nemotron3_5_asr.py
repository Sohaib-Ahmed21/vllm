# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal input processing for NVIDIA Nemotron 3.5 ASR."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
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

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
        )

    def get_feature_extractor(self, **kwargs: object) -> Any:
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
        audio_len = (max_mel_frames + 1) * feature_extractor.hop_length - 1
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

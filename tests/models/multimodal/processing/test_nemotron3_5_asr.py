# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BatchFeature, PreTrainedTokenizerBase

from vllm.model_executor.models.nemotron3_5_asr import (
    Nemotron3_5AsrDummyInputsBuilder,
    Nemotron3_5AsrMultiModalProcessor,
    Nemotron3_5AsrProcessingInfo,
    _get_subsampling_output_lengths,
)
from vllm.multimodal.inputs import MultiModalSharedField
from vllm.transformers_utils.processors import (
    Nemotron3_5AsrProcessor,
    NemotronAsrStreamingFeatureExtractor,
)


class _Tokenizer(PreTrainedTokenizerBase):
    def convert_tokens_to_ids(self, token):
        assert token == "<blank>"
        return 13087

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


class _MMConfig:
    enable_mm_embeds = False

    def merge_mm_processor_kwargs(self, kwargs):
        return dict(kwargs)

    def get_limit_per_prompt(self, modality):
        assert modality == "audio"
        return 1


class _HFProcessor:
    def __init__(self, valid_mel_frames: int):
        self.valid_mel_frames = valid_mel_frames
        self.feature_extractor = SimpleNamespace(sampling_rate=16000, hop_length=160)

    def __call__(self, audio, text, language="auto", **kwargs):
        del text, kwargs
        batch_size = len(audio)
        num_mel_frames = self.valid_mel_frames + 1
        attention_mask = torch.zeros(batch_size, num_mel_frames, dtype=torch.bool)
        attention_mask[:, : self.valid_mel_frames] = True

        return BatchFeature(
            {
                "input_features": torch.zeros(batch_size, num_mel_frames, 128),
                "attention_mask": attention_mask,
                "prompt_ids": torch.full(
                    (batch_size,), 10 if language == "en-US" else 101
                ),
                "num_lookahead_tokens": torch.tensor(3),
                "labels": torch.zeros(batch_size, 1, dtype=torch.long),
                "decoder_input_ids": torch.zeros(batch_size, 2, dtype=torch.long),
            }
        )


class _ProcessingContext:
    def __init__(self, valid_mel_frames: int, max_encoder_frames: int = 4):
        encoder_config = SimpleNamespace(
            max_position_embeddings=max_encoder_frames,
            subsampling_factor=8,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
        )
        self.model_config = SimpleNamespace(
            model="nvidia/nemotron-3.5-asr-streaming-0.6b",
            revision=None,
            max_model_len=4096,
            encoder_config={},
            hf_config=SimpleNamespace(
                blank_token_id=13087,
                encoder_config=encoder_config,
            ),
        )
        self.processor = _HFProcessor(valid_mel_frames)
        self.tokenizer = _Tokenizer()
        self.mm_config = _MMConfig()

    def get_tokenizer(self):
        return self.tokenizer

    def get_hf_config(self, typ=None):
        if typ is not None:
            assert isinstance(self.model_config.hf_config, typ)
        return self.model_config.hf_config

    def get_hf_processor(self, **kwargs):
        del kwargs
        return self.processor

    def get_mm_config(self):
        return self.mm_config

    def get_merged_mm_kwargs(self, kwargs):
        return self.mm_config.merge_mm_processor_kwargs(kwargs)

    def call_hf_processor(self, hf_processor, data, kwargs):
        merged_kwargs = self.get_merged_mm_kwargs(kwargs)
        merged_kwargs.setdefault("return_tensors", "pt")
        return hf_processor(**data, **merged_kwargs)


def _build_processor(valid_mel_frames: int):
    class _TestProcessingInfo(Nemotron3_5AsrProcessingInfo):
        def get_hf_config(self):
            return self.ctx.get_hf_config()

        def get_hf_processor(self, **kwargs):
            return self.ctx.get_hf_processor(**kwargs)

    info = _TestProcessingInfo(_ProcessingContext(valid_mel_frames))
    return Nemotron3_5AsrMultiModalProcessor(
        info,
        Nemotron3_5AsrDummyInputsBuilder(info),
    )


@pytest.mark.parametrize(
    ("valid_mel_frames", "expected_encoder_frames"),
    [(25, 4), (26, 5)],
)
def test_nemotron_processor_builds_encoder_decoder_contract(
    valid_mel_frames: int,
    expected_encoder_frames: int,
) -> None:
    processor = _build_processor(valid_mel_frames)
    mm_items = processor.info.parse_mm_data({"audio": np.zeros(1600, dtype=np.float32)})

    processed = processor(
        "",
        mm_items=mm_items,
        hf_processor_mm_kwargs={"language": "en-US"},
    )

    assert processed["prompt_token_ids"] == [13087]
    assert processed["encoder_prompt_token_ids"] == [0] * expected_encoder_frames
    assert processed["mm_placeholders"]["audio"][0].length == (expected_encoder_frames)

    audio_item = processed["mm_kwargs"]["audio"][0]
    assert audio_item["input_features"].data.shape == (
        valid_mel_frames + 1,
        128,
    )
    assert audio_item["attention_mask"].data.shape == (valid_mel_frames + 1,)
    assert audio_item["attention_mask"].data.sum().item() == valid_mel_frames
    assert audio_item["prompt_ids"].data.item() == 10
    assert audio_item["prompt_ids"].field.keep_on_cpu
    assert audio_item["num_lookahead_tokens"].data.item() == 3
    assert isinstance(
        audio_item["num_lookahead_tokens"].field,
        MultiModalSharedField,
    )
    assert audio_item["num_lookahead_tokens"].field.keep_on_cpu


def test_nemotron_dummy_audio_fills_encoder_capacity() -> None:
    info = _build_processor(valid_mel_frames=25).info
    builder = Nemotron3_5AsrDummyInputsBuilder(info)

    mm_data = builder.get_dummy_mm_data(
        seq_len=4096,
        mm_counts={"audio": 1},
        mm_options={},
    )

    (audio,) = mm_data["audio"]
    feature_extractor_info = info.get_feature_extractor()
    hop_length = feature_extractor_info.hop_length
    assert len(audio) // hop_length == 24
    assert (len(audio) + 1) // hop_length == 25

    feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)
    features = feature_extractor(
        audio,
        sampling_rate=feature_extractor_info.sampling_rate,
        return_tensors="pt",
    )
    encoder_config = info.get_hf_config().encoder_config
    physical_frames = _get_subsampling_output_lengths(
        torch.tensor([features["input_features"].shape[1]]),
        subsampling_factor=encoder_config.subsampling_factor,
        subsampling_conv_kernel_size=(encoder_config.subsampling_conv_kernel_size),
        subsampling_conv_stride=encoder_config.subsampling_conv_stride,
    )
    valid_frames = _get_subsampling_output_lengths(
        features["attention_mask"].sum(-1),
        subsampling_factor=encoder_config.subsampling_factor,
        subsampling_conv_kernel_size=(encoder_config.subsampling_conv_kernel_size),
        subsampling_conv_stride=encoder_config.subsampling_conv_stride,
    )

    assert physical_frames.item() == encoder_config.max_position_embeddings
    assert valid_frames.item() == encoder_config.max_position_embeddings


def test_nemotron_offline_feature_extractor_masks_extra_stft_frame() -> None:
    feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)

    output = feature_extractor(
        np.zeros(4040, dtype=np.float32),
        sampling_rate=16000,
        return_tensors="pt",
    )

    assert output["input_features"].shape == (1, 26, 128)
    assert output["attention_mask"].shape == (1, 26)
    assert output["attention_mask"].sum().item() == 25
    assert not output["attention_mask"][0, -1]
    assert torch.count_nonzero(output["input_features"][0, -1]).item() == 0


def test_nemotron_mel_filters_match_librosa_reference() -> None:
    librosa = pytest.importorskip("librosa")
    feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)

    expected = librosa.filters.mel(
        sr=feature_extractor.sampling_rate,
        n_fft=feature_extractor.n_fft,
        n_mels=feature_extractor.feature_size,
        fmin=0.0,
        fmax=feature_extractor.sampling_rate / 2,
        norm="slaney",
    )

    torch.testing.assert_close(
        feature_extractor.mel_filters,
        torch.from_numpy(expected),
        rtol=1e-6,
        atol=1e-8,
    )


def test_nemotron_feature_extractor_rejects_wrong_sampling_rate() -> None:
    feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)

    with pytest.raises(ValueError, match="Expected sampling rate 16000"):
        feature_extractor(
            np.zeros(1600, dtype=np.float32),
            sampling_rate=8000,
        )


def test_nemotron_processor_builds_real_audio_batch() -> None:
    processor = Nemotron3_5AsrProcessor(
        NemotronAsrStreamingFeatureExtractor(feature_size=128),
        _Tokenizer(),
    )

    output = processor(
        audio=[
            np.zeros(4040, dtype=np.float32),
            np.zeros(1600, dtype=np.float32),
        ],
        sampling_rate=16000,
        language=["en-US", "auto"],
    )

    assert output["input_features"].shape == (2, 26, 128)
    assert output["attention_mask"].sum(-1).tolist() == [25, 10]
    assert output["prompt_ids"].tolist() == [0, 101]
    assert output["num_lookahead_tokens"] == 3
    assert output["input_features"][1, 10:].count_nonzero().item() == 0


def test_nemotron_processor_accepts_flat_audio_list() -> None:
    processor = Nemotron3_5AsrProcessor(
        NemotronAsrStreamingFeatureExtractor(feature_size=128),
        _Tokenizer(),
    )

    output = processor(
        audio=[0.0] * 1600,
        sampling_rate=16000,
    )

    assert output["input_features"].shape == (1, 11, 128)
    assert output["attention_mask"].sum().item() == 10
    assert output["prompt_ids"].tolist() == [101]


def test_nemotron_processor_rejects_unsupported_language() -> None:
    processor = Nemotron3_5AsrProcessor(
        NemotronAsrStreamingFeatureExtractor(feature_size=128),
        _Tokenizer(),
    )

    with pytest.raises(ValueError, match="Unsupported Nemotron language"):
        processor(
            audio=np.zeros(1600, dtype=np.float32),
            sampling_rate=16000,
            language="not-a-language",
        )

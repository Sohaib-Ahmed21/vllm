# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from transformers import BatchFeature
from transformers.audio_utils import make_list_of_audio, mel_filter_bank
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin

logger = logging.getLogger(__name__)

_LOG_ZERO_GUARD_VALUE = 2**-24
_DEFAULT_PROMPT_DICTIONARY = {
    "en-US": 0,
    "en": 0,
    "en-GB": 1,
    "enGB": 1,
    "es-ES": 2,
    "esES": 2,
    "es-US": 3,
    "es": 3,
    "zh-CN": 4,
    "zh-ZH": 4,
    "zh-TW": 5,
    "hi-IN": 6,
    "hi": 6,
    "hi-HI": 6,
    "ar-AR": 7,
    "ar": 7,
    "fr-FR": 8,
    "fr": 8,
    "de-DE": 9,
    "de": 9,
    "ja-JP": 10,
    "ja-JA": 10,
    "ru-RU": 11,
    "ru": 11,
    "pt-BR": 12,
    "pt-PT": 13,
    "pt": 13,
    "ko-KR": 14,
    "ko": 14,
    "ko-KO": 14,
    "it-IT": 15,
    "it": 15,
    "nl-NL": 16,
    "nl": 16,
    "pl-PL": 17,
    "pl": 17,
    "tr-TR": 18,
    "tr": 18,
    "uk-UA": 19,
    "uk": 19,
    "ro-RO": 20,
    "ro": 20,
    "el-GR": 21,
    "el": 21,
    "cs-CZ": 22,
    "cs": 22,
    "hu-HU": 23,
    "hu": 23,
    "sv-SE": 24,
    "sv": 24,
    "da-DK": 25,
    "da": 25,
    "fi-FI": 26,
    "fi": 26,
    "no-NO": 27,
    "no": 27,
    "nb-NO": 103,
    "nb": 103,
    "nn-NO": 104,
    "nn": 104,
    "sk-SK": 28,
    "sk": 28,
    "hr-HR": 29,
    "hr": 29,
    "bg-BG": 30,
    "bg": 30,
    "lt-LT": 31,
    "lt": 31,
    "et-EE": 60,
    "et": 60,
    "lv-LV": 61,
    "lv": 61,
    "sl-SI": 62,
    "sl": 62,
    "th-TH": 32,
    "vi-VN": 33,
    "id-ID": 34,
    "ms-MY": 35,
    "bn-IN": 36,
    "ur-PK": 37,
    "fa-IR": 38,
    "ta-IN": 39,
    "te-IN": 40,
    "mr-IN": 41,
    "gu-IN": 42,
    "kn-IN": 43,
    "ml-IN": 44,
    "si-LK": 45,
    "ne-NP": 46,
    "km-KH": 47,
    "sw-KE": 48,
    "am-ET": 49,
    "ha-NG": 50,
    "zu-ZA": 51,
    "yo-NG": 52,
    "ig-NG": 53,
    "af-ZA": 54,
    "rw-RW": 55,
    "so-SO": 56,
    "ny-MW": 57,
    "ln-CD": 58,
    "or-KE": 59,
    "he-IL": 64,
    "ku-TR": 65,
    "az-AZ": 66,
    "ka-GE": 67,
    "hy-AM": 68,
    "uz-UZ": 69,
    "tg-TJ": 70,
    "ky-KG": 71,
    "qu-PE": 80,
    "ay-BO": 81,
    "gn-PY": 82,
    "nah-MX": 83,
    "mi-NZ": 96,
    "haw-US": 97,
    "sm-WS": 98,
    "to-TO": 99,
    "fr-CA": 100,
    "mt-MT": 102,
    "auto": 101,
}


class NemotronAsrStreamingFeatureExtractor(SequenceFeatureExtractor):
    """Extract Nemotron's 128-bin, pre-emphasized log-mel features."""

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 512,
        win_length: int = 400,
        preemphasis: float = 0.97,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis

        mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(mel_filters.T).to(torch.float32)

    def _torch_extract_fbank_features(
        self,
        waveform: torch.Tensor,
        *,
        device: torch.device,
        center: bool,
    ) -> torch.Tensor:
        waveform = waveform.to(device)
        window = torch.hann_window(self.win_length, periodic=False, device=device)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
            center=center,
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)).pow(2)
        mel_spec = self.mel_filters.to(device) @ magnitudes
        return torch.log(mel_spec + _LOG_ZERO_GUARD_VALUE).permute(0, 2, 1)

    @staticmethod
    def _as_mono_tensor(audio: object) -> torch.Tensor:
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim > 1:
            logger.warning(
                "Only mono-channel audio is supported; averaging audio channels."
            )
            waveform = waveform.mean(-1)
        return waveform

    def __call__(
        self,
        raw_speech: object,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool | None = None,
        padding: str | bool | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> BatchFeature:
        del return_attention_mask, kwargs
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}."
            )
        if sampling_rate is None:
            logger.warning(
                "Audio sampling_rate was not provided; assuming %s.",
                self.sampling_rate,
            )

        if isinstance(raw_speech, (np.ndarray, torch.Tensor)) and raw_speech.ndim <= 1:
            audios = [raw_speech]
        elif isinstance(raw_speech, (Sequence, np.ndarray, torch.Tensor)):
            audios = list(raw_speech)
        else:
            audios = [raw_speech]
        waveforms = [self._as_mono_tensor(audio) for audio in audios]
        if not waveforms:
            raise ValueError("At least one audio waveform is required.")

        lengths = torch.tensor([audio.numel() for audio in waveforms], dtype=torch.long)
        target_length = int(lengths.max().item())
        if isinstance(padding, str) and padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required when padding='max_length'.")
            target_length = max_length
        elif padding is False or padding is None:
            if len(waveforms) > 1 and len(set(lengths.tolist())) != 1:
                raise ValueError("Variable-length audio requires padding.")
        if truncation and max_length is not None:
            target_length = min(target_length, max_length)
        if pad_to_multiple_of:
            target_length = (
                (target_length + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of

        target_length = max(target_length, 1)
        padded = torch.full(
            (len(waveforms), target_length),
            self.padding_value,
            dtype=torch.float32,
        )
        for index, waveform in enumerate(waveforms):
            length = min(waveform.numel(), target_length)
            padded[index, :length] = waveform[:length]
            lengths[index] = length

        if self.preemphasis is not None:
            time_mask = torch.arange(target_length)[None, :] < lengths[:, None]
            padded = torch.cat(
                [padded[:, :1], padded[:, 1:] - self.preemphasis * padded[:, :-1]],
                dim=1,
            )
            padded = padded.masked_fill(~time_mask, 0.0)

        extract_device = torch.device(device)
        input_features = self._torch_extract_fbank_features(
            padded,
            device=extract_device,
            center=True,
        )
        feature_lengths = torch.div(
            lengths,
            self.hop_length,
            rounding_mode="floor",
        )
        attention_mask = (
            torch.arange(input_features.shape[1], device=extract_device)[None, :]
            < feature_lengths.to(extract_device)[:, None]
        )
        input_features = input_features.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        return BatchFeature(
            {
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


class Nemotron3_5AsrProcessor(ProcessorMixin):
    """HF-compatible audio processor for the Nemotron ASR checkpoint."""

    feature_extractor_class = "NemotronAsrStreamingFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        blank_token: str = "<blank>",
        supported_num_lookahead_tokens: list[int] | None = None,
        default_num_lookahead_tokens: int | None = None,
        prompt_dictionary: dict[str, int] | None = None,
        num_prompts: int = 128,
        **kwargs,
    ):
        del kwargs
        self.prompt_dictionary = prompt_dictionary or _DEFAULT_PROMPT_DICTIONARY
        self.num_prompts = num_prompts
        self.supported_num_lookahead_tokens = supported_num_lookahead_tokens or [
            3,
            0,
            6,
            13,
        ]
        self.default_num_lookahead_tokens = (
            default_num_lookahead_tokens
            if default_num_lookahead_tokens is not None
            else self.supported_num_lookahead_tokens[0]
        )
        self.blank_token = blank_token
        self.blank_token_id = tokenizer.convert_tokens_to_ids(blank_token)
        super().__init__(feature_extractor, tokenizer)

    def _resolve_prompt_ids(
        self, language: str | list[str], batch_size: int
    ) -> torch.Tensor:
        languages = [language] * batch_size if isinstance(language, str) else language
        if len(languages) != batch_size:
            raise ValueError("language must contain one entry per audio input.")
        prompt_ids = []
        for item in languages:
            prompt_id = self.prompt_dictionary.get(item)
            if prompt_id is None:
                prompt_id = self.prompt_dictionary.get(item.split("-", 1)[0])
            if prompt_id is None:
                raise ValueError(f"Unsupported Nemotron language: {item!r}")
            if not 0 <= prompt_id < self.num_prompts:
                raise ValueError(f"Prompt id {prompt_id} is outside num_prompts.")
            prompt_ids.append(prompt_id)
        return torch.tensor(prompt_ids, dtype=torch.long)

    def __call__(
        self,
        audio=None,
        text=None,
        sampling_rate: int | None = None,
        language: str | list[str] = "auto",
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        if audio is None:
            raise ValueError("Nemotron ASR requires audio input.")
        if isinstance(audio, (list, tuple)) and audio and np.isscalar(audio[0]):
            audio = np.asarray(audio, dtype=np.float32)
        audio_list = make_list_of_audio(audio)
        inputs = self.feature_extractor(
            audio_list,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors,
            **kwargs,
        )
        inputs["prompt_ids"] = self._resolve_prompt_ids(language, len(audio_list))
        inputs["num_lookahead_tokens"] = self.default_num_lookahead_tokens

        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=True,
                add_special_tokens=False,
            )
            inputs["labels"] = text_inputs["input_ids"]
            if isinstance(text, str):
                text = [text]
            decoder_text = [self.blank_token + item for item in text]
            inputs["decoder_input_ids"] = self.tokenizer(
                decoder_text,
                return_tensors=return_tensors,
                padding=True,
                add_special_tokens=False,
            )["input_ids"]
        return inputs

    @property
    def model_input_names(self):
        return self.feature_extractor.model_input_names + [
            "labels",
            "decoder_input_ids",
            "prompt_ids",
        ]


__all__ = [
    "NemotronAsrStreamingFeatureExtractor",
    "Nemotron3_5AsrProcessor",
]

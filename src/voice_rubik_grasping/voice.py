"""Offline voice command parsing and optional speech recognition."""

from __future__ import annotations

import os
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DEFAULT_ALIASES, load_yaml


@dataclass(frozen=True)
class ParsedCommand:
    """Structured result produced from one voice or text command."""

    raw_text: str
    target: Optional[str]
    confidence: float
    is_exit: bool = False


class VoiceCommandParser:
    """Map noisy spoken commands to open-vocabulary detector prompts."""

    def __init__(self, alias_path: str | Path = DEFAULT_ALIASES) -> None:
        aliases = load_yaml(alias_path)
        self.exit_commands = set(aliases.get("exit_commands", []))
        self.remove_words = tuple(aliases.get("remove_words", []))
        self.colors = dict(aliases.get("colors", {}))
        self.objects = dict(aliases.get("objects", {}))

    def parse(self, text: str | None) -> ParsedCommand:
        if not text:
            return ParsedCommand(raw_text="", target=None, confidence=0.0)

        raw_text = text.strip()
        lowered = raw_text.lower()
        if any(command.lower() in lowered for command in self.exit_commands):
            return ParsedCommand(
                raw_text=raw_text, target=None, confidence=1.0, is_exit=True
            )

        cleaned = raw_text
        for word in self.remove_words:
            cleaned = cleaned.replace(word, " ")
        cleaned = " ".join(cleaned.split())

        color = ""
        target = ""
        confidence = 0.0

        for source, mapped in sorted(self.colors.items(), key=lambda item: len(item[0]), reverse=True):
            if source in raw_text:
                color = mapped
                confidence += 0.3
                break

        for source, mapped in sorted(self.objects.items(), key=lambda item: len(item[0]), reverse=True):
            if source in raw_text:
                target = mapped
                confidence += 0.7
                break

        if target:
            if color and color not in target:
                target = f"{color} {target}"
            return ParsedCommand(
                raw_text=raw_text, target=target, confidence=min(confidence, 1.0)
            )

        if 0 < len(cleaned) < 32:
            return ParsedCommand(raw_text=raw_text, target=cleaned, confidence=0.3)

        return ParsedCommand(raw_text=raw_text, target=None, confidence=0.0)


class WhisperVoiceRecognizer:
    """Small Faster-Whisper wrapper used for the hardware voice demo."""

    def __init__(
        self,
        model_size: str = "small",
        sample_rate: int = 16_000,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )

    def record_audio(
        self,
        max_duration_s: float = 5.0,
        silence_threshold: float = 0.015,
        silence_duration_s: float = 0.8,
    ) -> Optional[np.ndarray]:
        import sounddevice as sd

        chunks: list[np.ndarray] = []
        silence_time = 0.0
        started = False

        def callback(indata, frames, _time_info, status) -> None:
            nonlocal silence_time, started
            if status:
                print(f"Audio status: {status}")
            chunks.append(indata.copy())
            volume = float(np.abs(indata).mean())
            if volume > silence_threshold:
                started = True
                silence_time = 0.0
            elif started:
                silence_time += frames / self.sample_rate

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=int(self.sample_rate * 0.1),
        ):
            start = time.time()
            while time.time() - start < max_duration_s:
                time.sleep(0.05)
                if started and silence_time > silence_duration_s:
                    break

        if not chunks or not started:
            return None
        return np.concatenate(chunks, axis=0).flatten()

    def transcribe(self, audio: np.ndarray) -> str:
        if self.model is None:
            self.load()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)

        try:
            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                audio_i16 = np.clip(audio, -1.0, 1.0)
                wav_file.writeframes((audio_i16 * 32767).astype(np.int16).tobytes())

            segments, _info = self.model.transcribe(
                str(temp_path), language="zh", beam_size=5, vad_filter=True
            )
            return "".join(segment.text for segment in segments).strip()
        finally:
            temp_path.unlink(missing_ok=True)

    def listen_once(self, parser: VoiceCommandParser) -> ParsedCommand:
        audio = self.record_audio()
        if audio is None:
            return ParsedCommand(raw_text="", target=None, confidence=0.0)
        return parser.parse(self.transcribe(audio))

from __future__ import annotations

from pathlib import Path


def save_audio(text: str, output_path: str = "outputs/response.mp3", language: str = "en"):
    if not text or not str(text).strip():
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from gtts import gTTS
        gTTS(text=str(text), lang=language).save(str(path))
        return str(path)
    except Exception as exc:
        print(f"TTS unavailable offline or failed: {exc}")
        return None


def speak(text: str, output_path: str = "outputs/response.mp3", language: str = "en"):
    return save_audio(text, output_path=output_path, language=language)

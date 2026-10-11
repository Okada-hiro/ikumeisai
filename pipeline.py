"""customer.wav -> transcription.txt / translation.txt -> translation.wav"""
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from openai import OpenAI

import config

_client = OpenAI(base_url=config.VLLM_BASE_URL, api_key="EMPTY", timeout=300)


def new_session_dir() -> Path:
    d = config.OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_customer_wav(sr: int, audio: np.ndarray, session: Path) -> Path:
    """Save mic input as mono float32 wav."""
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    path = session / "customer.wav"
    sf.write(path, audio.astype(np.float32), sr)
    return path


def _wav_16k_b64(path: Path) -> str:
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def parse_output(text: str) -> tuple[str, str]:
    data = json.loads(text)
    return data["transcript"].strip(), data["translation"].strip()


def transcribe_translate(wav_path: Path, session: Path) -> tuple[str, str, str]:
    resp = _client.chat.completions.create(
        model=config.VLLM_MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "audio_url",
                 "audio_url": {"url": "data:audio/wav;base64," + _wav_16k_b64(wav_path)}},
                {"type": "text", "text": config.PROMPT},
            ],
        }],
        response_format=config.RESPONSE_FORMAT,
        temperature=0.0,
        max_tokens=1024,
    )
    raw = resp.choices[0].message.content
    (session / "raw_output.txt").write_text(raw, encoding="utf-8")
    ja, en = parse_output(raw)
    (session / "transcription.txt").write_text(ja, encoding="utf-8")
    (session / "translation.txt").write_text(en, encoding="utf-8")
    return ja, en, raw


def synthesize(wav_path: Path, transcription: str, translation: str, session: Path) -> Path:
    """Voice clone via vLLM-Omni /v1/audio/speech (Qwen3-TTS Base)."""
    ref_b64 = base64.b64encode(wav_path.read_bytes()).decode()
    resp = requests.post(
        f"{config.TTS_BASE_URL}/audio/speech",
        json={
            "model": config.TTS_MODEL_NAME,
            "input": translation,
            "task_type": "Base",
            "language": "English",
            "ref_audio": "data:audio/wav;base64," + ref_b64,
            "ref_text": transcription,
            "response_format": "wav",
        },
        timeout=600,
    )
    resp.raise_for_status()
    out = session / "translation.wav"
    out.write_bytes(resp.content)
    return out


def run(wav_path: Path, session: Path) -> dict:
    ja, en, raw = transcribe_translate(wav_path, session)
    out_wav = synthesize(wav_path, ja, en, session)
    return {"transcription": ja, "translation": en, "raw": raw, "translation_wav": out_wav}


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1])
    s = new_session_dir()
    audio, sr = sf.read(src, dtype="float32")
    print(run(save_customer_wav(sr, audio, s), s))

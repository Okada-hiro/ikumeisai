"""FLEURS ja_jp の3サンプルでパイプライン全体を動作確認する。
python test_samples.py  (両サーバー起動済みであること)
"""
import json
import time
from pathlib import Path

import soundfile as sf

import pipeline

TEST_DIR = Path(__file__).resolve().parent / "test_audio"

for m in json.loads((TEST_DIR / "samples.json").read_text()):
    session = pipeline.new_session_dir()
    audio, sr = sf.read(TEST_DIR / m["file"], dtype="float32")
    wav = pipeline.save_customer_wav(sr, audio, session)
    t0 = time.time()
    ja, en, raw = pipeline.transcribe_translate(wav, session)
    t1 = time.time()
    out = pipeline.synthesize(wav, ja, en, session)
    t2 = time.time()
    out_audio, out_sr = sf.read(out)
    print(f"=== {m['file']} ({m['duration_s']}s) -> {session.name}")
    print(f"  正解   : {m['reference']}")
    print(f"  文字起こし: {ja}")
    print(f"  英訳   : {en}")
    print(f"  raw    : {raw!r}")
    print(f"  時間   : 文字起こし+英訳 {t1 - t0:.1f}s / TTS {t2 - t1:.1f}s"
          f" (出力 {len(out_audio) / out_sr:.1f}s)")

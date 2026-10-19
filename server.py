"""ikumeisai アプリサーバー: UI (ui/) とストリーミング API (:7861)。

起動するとモデルサーバー (TTS :8001 → Omni :8000) を未起動なら起動し、テスト音声でウォームアップしてから
/api/status を ready にする。UI はそれまで起動画面を表示する。

POST /api/process (body: audio/wav) は NDJSON で次のイベントを流す (ui/assets/api-processor.js が受信):
  ?lang=en|fr|zh|de|ru で翻訳先を選ぶ (config.PROMPTS / config.TTS_LANGUAGES)
  {"type": "transcript"|"translation", "text": <その時点の全文>, "done": bool}  ← Omni の生成トークンごと
  {"type": "status", "message": ...}
  {"type": "audio", "url": "/api/audio/<session>/translation.wav"}
  {"type": "timing", ...} / {"type": "error", "message": ...}
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import re
import subprocess
import time
from pathlib import Path

import httpx
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
import pipeline

UI_DIR = config.ROOT / "ui"
LOG_DIR = Path.home() / "logs"
WARMUP_WAV = config.ROOT / "test_audio" / "sample1.wav"
WARMUP_ROUNDS = 2  # 1 回目はカーネルの初回コンパイル等で遅いので 2 回流す

# (id, 起動スクリプト, ポート, ログ, 表示名) — TTS を先に起動 (小さい方が先に GPU メモリを確保する)
SERVERS = [
    ("tts", "serve_tts.sh", 8001, "tts_server.log", "音声合成モデル (Qwen3-TTS) を読み込んでいます"),
    ("omni", "serve_vllm.sh", 8000, "vllm_server.log", "文字起こし・翻訳モデル (Qwen3-Omni) を読み込んでいます"),
]

STATE: dict = {
    "phase": "starting",  # starting | loading | warming | ready | error
    "started": time.time(),
    "error": None,
    "steps": [{"id": s[0], "label": s[4], "status": "pending"} for s in SERVERS]
    + [{"id": "warmup", "label": "ウォームアップしています", "status": "pending"}],
}
_procs: dict[str, subprocess.Popen] = {}
_lock = asyncio.Lock()


def _step(step_id: str, status: str, detail: str | None = None) -> None:
    for s in STATE["steps"]:
        if s["id"] == step_id:
            if status == "active" and s["status"] != "active":
                s["since"] = time.time()
            s["status"] = status
            if status == "done" and "since" in s:
                s["seconds"] = round(time.time() - s["since"], 1)
            if detail is not None:
                s["detail"] = detail


async def _up(port: int) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            return (await c.get(f"http://127.0.0.1:{port}/v1/models")).status_code == 200
    except httpx.HTTPError:
        return False


def _log_tail(name: str, n: int = 15) -> str:
    try:
        return "\n".join((LOG_DIR / name).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


async def _ensure_server(sid: str, script: str, port: int, log: str) -> None:
    if await _up(port):
        return
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / log, "w") as fh:
        _procs[sid] = subprocess.Popen(
            [str(config.ROOT / script)], cwd=config.ROOT, stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    while not await _up(port):
        if _procs[sid].poll() is not None:
            raise RuntimeError(f"{script} が異常終了しました (~/logs/{log})\n{_log_tail(log)}")
        await asyncio.sleep(2)


async def boot() -> None:
    try:
        STATE["phase"] = "loading"
        for sid, script, port, log, _ in SERVERS:
            _step(sid, "active")
            await _ensure_server(sid, script, port, log)
            _step(sid, "done")
        STATE["phase"] = "warming"
        _step("warmup", "active")
        session = config.OUTPUT_DIR / "_warmup"
        session.mkdir(parents=True, exist_ok=True)
        (session / "customer.wav").write_bytes(WARMUP_WAV.read_bytes())
        for i in range(WARMUP_ROUNDS):
            _step("warmup", "active", f"{i + 1} / {WARMUP_ROUNDS}")
            async for ev in process_stream(session):
                if ev["type"] == "error":
                    raise RuntimeError(ev["message"])
        _step("warmup", "done", "")
        STATE["phase"] = "ready"
        STATE["ready_at"] = time.time()
    except Exception as e:  # noqa: BLE001 — 起動画面にそのまま表示する
        STATE["phase"] = "error"
        STATE["error"] = str(e)
        for s in STATE["steps"]:
            if s["status"] == "active":
                s["status"] = "error"


# ---------------------------------------------------------------- streaming JSON

_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}


def partial_field(text: str, key: str) -> tuple[str | None, bool]:
    """生成途中の JSON テキストから文字列フィールド ``key`` の値を取り出す。(値, 閉じたか)。"""
    k = text.find(f'"{key}"')
    if k < 0:
        return None, False
    i, n = k + len(key) + 2, len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    if i >= n or text[i] != ":":
        return None, False
    i += 1
    while i < n and text[i] in " \t\r\n":
        i += 1
    if i >= n or text[i] != '"':
        return None, False
    i += 1
    out: list[str] = []
    while i < n:
        c = text[i]
        if c == '"':
            return "".join(out), True
        if c == "\\":
            if i + 1 >= n:
                break  # エスケープの途中で切れている
            e = text[i + 1]
            if e == "u":
                if i + 6 > n:
                    break
                code = int(text[i + 2 : i + 6], 16)
                if 0xD800 <= code < 0xDC00:  # サロゲートペア
                    if i + 12 > n:
                        break
                    if text[i + 6 : i + 8] == "\\u":
                        low = int(text[i + 8 : i + 12], 16)
                        out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                        i += 12
                        continue
                out.append(chr(code))
                i += 6
                continue
            out.append(_ESC.get(e, e))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out), False


# ---------------------------------------------------------------- VAD

MIN_SPEECH_S = 0.3
_vad = None


def speech_seconds(audio, sr: int) -> float:
    """Silero VAD (ONNX, CPU) で発話と判定された秒数。音量ではなく声らしさで判定する。"""
    global _vad
    import numpy as np
    import importlib.util

    import onnxruntime as ort
    import soxr

    if _vad is None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        # silero_vad を import すると torch を読み込むので、パッケージ内の ONNX ファイルだけを使う
        pkg = Path(importlib.util.find_spec("silero_vad").submodule_search_locations[0])
        _vad = ort.InferenceSession(str(pkg / "data" / "silero_vad.onnx"),
                                    sess_options=opts, providers=["CPUExecutionProvider"])
    x = (audio if sr == 16000 else soxr.resample(audio, sr, 16000)).astype(np.float32)
    state = np.zeros((2, 1, 128), np.float32)
    ctx = np.zeros((1, 64), np.float32)  # 前のチャンク末尾 64 サンプル (公式 OnnxWrapper と同じ)
    voiced = 0
    for i in range(0, len(x) - 511, 512):
        chunk = x[None, i : i + 512]
        prob, state = _vad.run(None, {"input": np.concatenate([ctx, chunk], 1), "state": state,
                                      "sr": np.array(16000, np.int64)})
        ctx = chunk[:, -64:]
        voiced += float(prob[0, 0]) >= 0.5
    return voiced * 512 / 16000


# ---------------------------------------------------------------- pipeline

async def process_stream(session: Path, lang: str = "en"):
    """session/customer.wav → Omni (トークンごとにイベント) → TTS。イベント dict を yield する。

    lang: 翻訳先 (config.PROMPTS のキー)。プロンプトと TTS の language を切り替える。
    """
    wav = session / "customer.wav"
    t0 = time.time()
    audio_b64 = await asyncio.to_thread(pipeline._wav_16k_b64, wav)
    body = {
        "model": config.VLLM_MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + audio_b64}},
            {"type": "text", "text": config.PROMPTS[lang]},
        ]}],
        "response_format": config.RESPONSE_FORMAT,
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": True,
    }
    raw, sent = "", {"transcript": (None, False), "translation": (None, False)}
    t_first = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
        async with client.stream("POST", f"{config.VLLM_BASE_URL}/chat/completions", json=body) as r:
            if r.status_code != 200:
                yield {"type": "error", "message": f"Omni エラー {r.status_code}: {(await r.aread())[:300]!r}"}
                return
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                delta = (choices[0].get("delta") or {}).get("content") if choices else None
                if not delta:
                    continue
                t_first = t_first or time.time()
                raw += delta
                for key in ("transcript", "translation"):
                    cur = partial_field(raw, key)
                    if cur[0] is not None and cur != sent[key]:
                        sent[key] = cur
                        yield {"type": key, "text": cur[0], "done": cur[1]}
    t_llm = time.time()
    try:
        ja, en = pipeline.parse_output(raw)
    except (ValueError, KeyError) as e:
        yield {"type": "error", "message": f"文字起こし結果を解析できませんでした: {e}"}
        return
    for key, val in (("transcript", ja), ("translation", en)):
        if sent[key] != (val, True):
            yield {"type": key, "text": val, "done": True}
    (session / "raw_output.txt").write_text(raw, encoding="utf-8")
    (session / "transcription.txt").write_text(ja, encoding="utf-8")
    (session / "translation.txt").write_text(en, encoding="utf-8")

    yield {"type": "status", "message": "あなたの声で音声を生成しています"}
    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post(f"{config.TTS_BASE_URL}/audio/speech", json={
            "model": config.TTS_MODEL_NAME,
            "input": en,
            "task_type": "Base",
            "language": config.TTS_LANGUAGES[lang],
            "ref_audio": "data:audio/wav;base64," + base64.b64encode(wav.read_bytes()).decode(),
            # 中国語などは参照の書き起こしを使わず声の特徴だけで (config.TTS_XVECTOR_ONLY)
            **({"x_vector_only_mode": True} if lang in config.TTS_XVECTOR_ONLY else {"ref_text": ja}),
            "response_format": "wav",
        })
    if r.status_code != 200:
        yield {"type": "error", "message": f"音声合成エラー {r.status_code}: {r.text[:300]}"}
        return
    (session / "translation.wav").write_bytes(r.content)
    t_tts = time.time()
    yield {"type": "audio", "url": f"/api/audio/{session.name}/translation.wav"}
    yield {"type": "timing", "first_token_s": round((t_first or t_llm) - t0, 2),
           "transcribe_translate_s": round(t_llm - t0, 2), "tts_s": round(t_tts - t_llm, 2)}


# ---------------------------------------------------------------- app

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(boot())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/api/status")
async def status():
    now = time.time()
    steps = [dict(s, elapsed=round(now - s["since"], 1) if s["status"] == "active" else s.get("seconds"))
             for s in STATE["steps"]]
    return JSONResponse({"phase": STATE["phase"], "error": STATE["error"], "steps": steps,
                         "elapsed": round(now - STATE["started"], 1)}, headers={"Cache-Control": "no-store"})


@app.post("/api/process")
async def process(request: Request):
    if STATE["phase"] != "ready":
        raise HTTPException(503, "モデルの準備中です")
    lang = request.query_params.get("lang", "en")
    if lang not in config.PROMPTS:
        raise HTTPException(400, f"対応していない言語です: {lang}")
    data = await request.body()
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"音声を読み込めませんでした: {e}") from e
    if len(audio) < 0.3 * sr:
        raise HTTPException(400, "録音が短すぎます")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if await asyncio.to_thread(speech_seconds, audio, sr) < MIN_SPEECH_S:  # マイク未接続・無音・雑音のみ
        raise HTTPException(400, "声が検出されませんでした。マイクの接続を確認して、もう一度話してください")
    session = pipeline.new_session_dir()
    pipeline.save_customer_wav(sr, audio, session)

    async def events():
        async with _lock:  # GPU は 1 組のモデルを共有するので 1 件ずつ処理する
            try:
                async for ev in process_stream(session, lang):
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
            except Exception as e:  # noqa: BLE001
                yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/audio/{session}/{name}")
async def audio(session: str, name: str):
    if not re.fullmatch(r"[0-9-]+", session) or name != "translation.wav":
        raise HTTPException(404)
    path = config.OUTPUT_DIR / session / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host=config.APP_HOST, port=config.APP_PORT, log_level="info")

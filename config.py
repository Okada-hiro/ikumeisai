from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

OMNI_MODEL_PATH = HOME / "models" / "Qwen3-Omni-30B-A3B-Instruct"
TTS_MODEL_PATH = HOME / "models" / "Qwen3-TTS-12Hz-1.7B-Base"

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_MODEL_NAME = "qwen3-omni"

TTS_BASE_URL = "http://127.0.0.1:8001/v1"
TTS_MODEL_NAME = "qwen3-tts"

APP_HOST = "127.0.0.1"  # UI はこのマシンの Firefox か SSH トンネル経由で開く
APP_PORT = 7861

PROMPT = (
    "Transcript and translate the following Japanese audio into English as a json form. "
    'For example, your output is "{transcript: 今日は晴れです。, translation: It\'s sunny today.}"'
)

# vLLM structured output: forces valid JSON with exactly these keys
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "transcription_translation",
        "schema": {
            "type": "object",
            "properties": {
                "transcript": {"type": "string"},
                "translation": {"type": "string"},
            },
            "required": ["transcript", "translation"],
        },
    },
}

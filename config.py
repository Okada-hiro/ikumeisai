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

# 他言語版 (まだ未使用、いずれも Qwen3-TTS の対応言語): PROMPT の "English" と例文の訳 "It's sunny today." だけを置き換えたもの。
# キーは翻訳先言語。PROMPTS["en"] は PROMPT と同一。
_PROMPT_TEMPLATE = (
    "Transcript and translate the following Japanese audio into {language} as a json form. "
    'For example, your output is "{{transcript: 今日は晴れです。, translation: {example}}}"'
)
TARGET_LANGUAGES = {  # code: (プロンプト中の言語名, 例文の訳, Qwen3-TTS の language 値)
    "en": ("English", "It's sunny today.", "English"),
    "fr": ("French", "Il fait beau aujourd'hui.", "French"),
    "zh": ("Chinese", "今天是晴天。", "Chinese"),
    "de": ("German", "Heute ist es sonnig.", "German"),
    "ru": ("Russian", "Сегодня солнечно.", "Russian"),
}
PROMPTS = {code: _PROMPT_TEMPLATE.format(language=lang, example=ex)
           for code, (lang, ex, _) in TARGET_LANGUAGES.items()}
TTS_LANGUAGES = {code: tts for code, (_, _, tts) in TARGET_LANGUAGES.items()}
assert PROMPTS["en"] == PROMPT

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

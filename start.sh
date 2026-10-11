#!/usr/bin/env bash
# TTS (vLLM-Omni, :8001) と Omni (vLLM, :8000) を未起動なら起動し、準備完了を待って UI (:7861) を起動
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ~/logs

start_server() {  # name script port log
  local name=$1 script=$2 port=$3 log=$4
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    echo "[start] ${name} は起動済み (:${port})"
    return
  fi
  echo "[start] ${name} を起動します (ログ: ${log})"
  nohup "./${script}" > "${log}" 2>&1 &
  echo $! > "/tmp/ikumeisai_${name}.pid"
}

wait_server() {  # name port
  local name=$1 port=$2 pid
  pid=$(cat "/tmp/ikumeisai_${name}.pid" 2>/dev/null || true)
  until curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      echo "[start] ${name} が異常終了しました。ログを確認してください" >&2
      exit 1
    fi
    sleep 10
  done
  echo "[start] ${name} 準備完了 (:${port})"
}

# TTS を先に起動 (小さい方が先に GPU メモリを確保する)
start_server tts serve_tts.sh 8001 ~/logs/tts_server.log
wait_server tts 8001
start_server omni serve_vllm.sh 8000 ~/logs/vllm_server.log
wait_server omni 8000

source ~/venvs/app/bin/activate
exec python app.py

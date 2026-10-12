#!/usr/bin/env bash
# アプリ (UI・アプリサーバー) とモデルサーバー (Omni / TTS) をすべて終了して GPU メモリを解放する
set -u
pkill -f "[i]kumeisai-profile" || true
pkill -INT -f "[v]envs/app/bin/python server.py" || true
pkill -INT -f "bin/[v]llm serve" || true
for _ in $(seq 1 30); do
  pgrep -f "bin/[v]llm serve|[v]envs/app/bin/python server.py" >/dev/null || exit 0
  sleep 1
done
pkill -KILL -f "bin/[v]llm serve" || true

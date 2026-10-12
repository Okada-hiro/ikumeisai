#!/usr/bin/env bash
# ターミナルから起動する場合 (デスクトップからは launch_app.sh)。アプリサーバー (:7861) が
# TTS (:8001) → Omni (:8000) を起動・ウォームアップし、UI は http://localhost:7861/ で開く。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ~/logs
exec ~/venvs/app/bin/python server.py

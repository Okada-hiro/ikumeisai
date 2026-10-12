#!/usr/bin/env bash
# ワンクリック起動 (デスクトップの「文字起こし&翻訳AI」から呼ばれる。Mac 版は --server-only で呼ぶ):
# アプリサーバー (:7861) を未起動なら起動し、専用プロファイルの Firefox を全画面 (kiosk) で開く。
# モデルの読み込みとウォームアップはアプリサーバーが行い、その間 UI は起動画面を表示する。
set -u
cd "$(dirname "$0")"
mkdir -p ~/logs
PORT=7861
PROFILE="$HOME/snap/firefox/common/ikumeisai-profile"  # snap 版 Firefox が読める場所

if ! curl -sf "http://127.0.0.1:${PORT}/api/status" >/dev/null; then
  setsid nohup ~/venvs/app/bin/python server.py >> ~/logs/app.log 2>&1 < /dev/null &
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${PORT}/api/status" >/dev/null && break
    sleep 0.5
  done
fi

# --server-only: 画面は開かない (Mac 側アプリが SSH 越しに呼ぶ)
if [[ "${1:-}" == "--server-only" ]]; then
  exit 0
fi

# すでに開いている場合は二重に開かない
if pgrep -f "[i]kumeisai-profile" >/dev/null; then
  exit 0
fi
exec firefox --profile "$PROFILE" --no-remote --kiosk "http://localhost:${PORT}/"

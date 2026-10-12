#!/bin/bash
# Mac 版ワンクリック起動: EVO-X2 のアプリサーバーを画面なしで起動 (モデル読込・ウォームアップは EVO-X2 側)、
# SSH トンネル (Mac localhost:7861 → EVO-X2 127.0.0.1:7861) を張り、Mac のブラウザで UI を開く。
# 録音は Mac のマイク、音声はトンネル (SSH) で EVO-X2 に送られ、合成音声は Mac で再生される。
#   ikumeisai_mac.sh        起動
#   ikumeisai_mac.sh stop   EVO-X2 のアプリとモデルを終了し、トンネルを閉じる
HOST="${IKUMEISAI_HOST:-sugakuka@192.168.1.12}"
REMOTE_DIR="${IKUMEISAI_REMOTE_DIR:-~/workspace/ikumeisai}"
PORT=7861
LOCAL_PORT="${IKUMEISAI_LOCAL_PORT:-7861}"
URL="http://localhost:${LOCAL_PORT}/"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
TUNNEL_PAT="[s]sh .*-L ${LOCAL_PORT}:127.0.0.1:${PORT}"

alert() { osascript -e "display alert \"文字起こし&翻訳AI\" message \"$1\" as critical" >/dev/null 2>&1; }

if [[ "${1:-}" == "stop" ]]; then
  ssh "${SSH_OPTS[@]}" "$HOST" "${REMOTE_DIR}/stop_app.sh" || alert "EVO-X2 (${HOST}) に接続できませんでした。"
  pkill -f "$TUNNEL_PAT" || true
  pkill -f "[i]kumeisai-chrome" || true
  exit 0
fi

if ! ssh "${SSH_OPTS[@]}" "$HOST" "${REMOTE_DIR}/launch_app.sh --server-only"; then
  alert "EVO-X2 (${HOST}) に接続できませんでした。電源とネットワークを確認してください。"
  exit 1
fi

if ! curl -sf "http://127.0.0.1:${LOCAL_PORT}/api/status" >/dev/null; then
  pkill -f "$TUNNEL_PAT" || true  # 切れかけの古いトンネルを掃除
  if ! ssh "${SSH_OPTS[@]}" -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
       -L "${LOCAL_PORT}:127.0.0.1:${PORT}" "$HOST"; then
    alert "SSH トンネルを張れませんでした (Mac のポート ${LOCAL_PORT} が使用中の可能性があります)。"
    exit 1
  fi
fi

CHROME="/Applications/Google Chrome.app"
if [[ -d "$CHROME" ]]; then
  # 専用プロファイルのアプリウィンドウ: マイク許可ダイアログと自動再生ブロックを省く (このアプリ専用)
  open -na "$CHROME" --args --app="$URL" \
    --user-data-dir="$HOME/Library/Application Support/ikumeisai-chrome" \
    --use-fake-ui-for-media-stream --autoplay-policy=no-user-gesture-required \
    --no-first-run --no-default-browser-check
else
  open "$URL"
fi

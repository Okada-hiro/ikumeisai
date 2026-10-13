#!/bin/bash
# Mac 版ワンクリック起動: EVO-X2 を同じ Wi-Fi 上で探して SSH 接続し、アプリサーバーを画面なしで起動
# (モデル読込・ウォームアップは EVO-X2 側)、SSH トンネル (Mac localhost:7861 → EVO-X2 127.0.0.1:7861) を張って
# Mac のブラウザで UI を開く。録音は Mac のマイク、音声はトンネル (SSH) で EVO-X2 に送られる。
#   ikumeisai_mac.sh        起動
#   ikumeisai_mac.sh stop   EVO-X2 のアプリとモデルを終了し、トンネルを閉じる
# 前提: EVO-X2 が Ubuntu で起動済み (ログイン不要) で Mac と同じ Wi-Fi にいること。
#       EVO-X2 側で一度 ./setup_remote_access.sh を実行しておくこと (SSH・mDNS・Wi-Fi をログイン前から有効化)。
USER_AT="${IKUMEISAI_USER:-sugakuka}"
# 試す順: 前回つながった宛先 → 固定 IP → mDNS 名 (DHCP で IP が変わっても同じ Wi-Fi なら見つかる)
HOSTS="${IKUMEISAI_HOSTS:-192.168.1.12 sugakuka-NucBox-EVO-X2.local}"
REMOTE_DIR="${IKUMEISAI_REMOTE_DIR:-~/workspace/ikumeisai}"
WAIT_S="${IKUMEISAI_WAIT_S:-240}"  # 電源を入れた直後でも待てるように
PORT=7861
LOCAL_PORT="${IKUMEISAI_LOCAL_PORT:-7861}"
URL="http://localhost:${LOCAL_PORT}/"
STATE_DIR="$HOME/Library/Application Support/ikumeisai"
LAST_HOST_FILE="$STATE_DIR/last_host"
# どの宛先名で入っても、known_hosts に登録済みの EVO-X2 のホスト鍵で検証する
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=4 -o HostKeyAlias=192.168.1.12 -o ServerAliveInterval=30)
TUNNEL_PAT="[s]sh .*-L ${LOCAL_PORT}:127.0.0.1:${PORT}"

notify() { osascript -e "display notification \"$1\" with title \"文字起こし&翻訳AI\"" >/dev/null 2>&1; }
alert() { osascript -e "display alert \"文字起こし&翻訳AI\" message \"$1\" as critical" >/dev/null 2>&1; }
mac_ssid() {  # 位置情報の許可がないと macOS は SSID を <redacted> にする
  local s
  s=$(ipconfig getsummary en0 2>/dev/null | awk -F ' SSID : ' '/ SSID : / {print $2; exit}')
  [[ -n "$s" && "$s" != *redacted* ]] && echo "$s"
}

# EVO-X2 を探す: つながった宛先を標準出力に出す
find_host() {  # wait_seconds
  local deadline=$(( $(date +%s) + $1 )) notified=0 last=""
  [[ -f "$LAST_HOST_FILE" ]] && last=$(cat "$LAST_HOST_FILE")
  while :; do
    for h in $last $HOSTS; do
      if ssh "${SSH_OPTS[@]}" "${USER_AT}@${h}" true 2>/dev/null; then
        mkdir -p "$STATE_DIR"; echo "$h" > "$LAST_HOST_FILE"
        echo "$h"; return 0
      fi
    done
    if (( ! notified )); then
      notify "EVO-X2 に接続しています… (電源を入れた直後なら数分かかります)"
      notified=1
    fi
    (( $(date +%s) >= deadline )) && return 1
    sleep 5
  done
}

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "$TUNNEL_PAT" || true
  pkill -f "[i]kumeisai-chrome" || true
  if HOST=$(find_host 10); then
    ssh "${SSH_OPTS[@]}" "${USER_AT}@${HOST}" "${REMOTE_DIR}/stop_app.sh"
  fi
  exit 0
fi

if ! HOST=$(find_host "$WAIT_S"); then
  ssid=$(mac_ssid)
  alert "EVO-X2 に接続できませんでした。次を確認してください。
・EVO-X2 の電源が入り、Ubuntu で起動している (ログインは不要)
・EVO-X2 と Mac が同じ Wi-Fi にいる${ssid:+ (この Mac: ${ssid})}
試した宛先: ${HOSTS}"
  exit 1
fi

if ! ssh "${SSH_OPTS[@]}" "${USER_AT}@${HOST}" "${REMOTE_DIR}/launch_app.sh --server-only"; then
  alert "EVO-X2 (${HOST}) でアプリを起動できませんでした。EVO-X2 の ~/logs/app.log を確認してください。"
  exit 1
fi

if ! curl -sf "http://127.0.0.1:${LOCAL_PORT}/api/status" >/dev/null; then
  pkill -f "$TUNNEL_PAT" || true  # 切れた古いトンネル (IP が変わった等) を掃除
  if ! ssh "${SSH_OPTS[@]}" -f -N -o ExitOnForwardFailure=yes \
       -L "${LOCAL_PORT}:127.0.0.1:${PORT}" "${USER_AT}@${HOST}"; then
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

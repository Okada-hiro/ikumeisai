#!/usr/bin/env bash
# Mac のアプリ (mac/ikumeisai_mac.sh) がクリックだけで入れるよう、EVO-X2 が起動した時点で (ログイン不要)
# SSH・mDNS (<hostname>.local)・Wi-Fi が有効になるようにする。一度だけ実行 (sudo を使う)。
set -euo pipefail

echo "== SSH サーバー (起動時に有効化)"
sudo systemctl enable --now ssh.service >/dev/null 2>&1 || true
sudo systemctl enable --now ssh.socket >/dev/null 2>&1 || true
systemctl is-active ssh.service ssh.socket | paste -sd' ' | sed 's/^/  ssh.service ssh.socket: /'

echo "== mDNS (avahi): Mac から $(hostname).local で見つけられるようにする"
sudo systemctl enable --now avahi-daemon >/dev/null
echo "  avahi-daemon: $(systemctl is-active avahi-daemon)"

echo "== Wi-Fi: 全ユーザー共通 (ログイン前に接続)・自動接続・省電力オフ"
nmcli -t -f NAME,TYPE connection show | while IFS=: read -r name type; do
  [[ "$type" == "802-11-wireless" ]] || continue
  sudo nmcli connection modify "$name" connection.permissions "" connection.autoconnect yes \
    802-11-wireless.powersave 2
  flags=$(sudo nmcli -g 802-11-wireless-security.psk-flags connection show "$name" 2>/dev/null || echo "?")
  if [[ "$flags" == "0" || "$flags" == "0 (none)" || -z "$flags" ]]; then
    echo "  $name: OK (パスワードはシステムに保存済み)"
  else
    echo "  $name: 注意 — パスワードが個人のキーリングにある (psk-flags=$flags) のでログイン前に接続できない。"
    echo "         sudo nmcli connection modify \"$name\" 802-11-wireless-security.psk-flags 0 802-11-wireless-security.psk <パスワード>"
  fi
done

if command -v ufw >/dev/null && sudo ufw status | grep -q "Status: active"; then
  echo "== ファイアウォール (ufw): SSH と mDNS を許可"
  sudo ufw allow 22/tcp >/dev/null
  sudo ufw allow 5353/udp >/dev/null
fi

echo "== 現在の接続先"
ip -4 -o addr show scope global | awk '{print "  " $2 ": " $4}'
echo "  Mac からは ssh sugakuka@$(hostname).local でも入れる"

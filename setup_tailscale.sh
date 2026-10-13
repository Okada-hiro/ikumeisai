#!/usr/bin/env bash
# EVO-X2 を Tailscale (WireGuard のオーバーレイ VPN) に参加させ、Mac と同じ Wi-Fi にいなくても
# EVO-X2 が起動さえしていれば ssh (と Mac 版アプリ) で入れるようにする。一度だけ実行 (sudo を使う)。
#
# 大学のルーターのポート開放は不要 (EVO-X2 から外向きに接続するだけ)。インターネットに公開されるポートもない。
# 初回は最後に表示される URL をブラウザで開き、Mac と同じ Tailscale アカウントでログインする。
# ログイン後、管理画面 (https://login.tailscale.com/admin/machines) で evo-x2 の「Disable key expiry」を
# 有効にしておくと、180 日ごとの再ログインが要らない。
set -euo pipefail

if ! command -v tailscale >/dev/null; then
  # 公式 apt リポジトリ (https://tailscale.com/download/linux/ubuntu-2404 の手順と同じ)
  . /etc/os-release
  curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/${VERSION_CODENAME}.noarmor.gpg" \
    | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
  curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/${VERSION_CODENAME}.tailscale-keyring.list" \
    | sudo tee /etc/apt/sources.list.d/tailscale.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq tailscale
fi
sudo systemctl enable --now tailscaled   # 起動時に自動で接続 (ログイン不要)
sudo tailscale up --hostname=evo-x2      # 初回はログイン用 URL が表示される
echo "Tailscale IP: $(tailscale ip -4)"
echo "Mac (同じアカウントで Tailscale にログイン済み) から: ssh sugakuka@evo-x2"

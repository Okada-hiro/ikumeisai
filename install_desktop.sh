#!/usr/bin/env bash
# デスクトップとアプリ一覧に「文字起こし&翻訳AI」を登録し、専用 Firefox プロファイルを用意する (一度だけ実行)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
chmod +x "$ROOT/launch_app.sh" "$ROOT/stop_app.sh"

# 専用プロファイル: マイク許可・自動再生を事前に許可 (localhost のこのアプリ専用)、初回ダイアログ類を無効化
PROFILE="$HOME/snap/firefox/common/ikumeisai-profile"
mkdir -p "$PROFILE"
cat > "$PROFILE/user.js" <<'EOF'
user_pref("permissions.default.microphone", 1);
user_pref("media.navigator.permission.disabled", true);
user_pref("media.autoplay.default", 0);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("trailhead.firstrun.didSeeAboutWelcome", true);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("startup.homepage_welcome_url", "");
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("browser.translations.automaticallyPopup", false);
EOF

ENTRY="$HOME/.local/share/applications/ikumeisai.desktop"
mkdir -p "$(dirname "$ENTRY")"
cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=文字起こし&翻訳AI
Comment=日本語の音声を文字起こし・英訳し、あなたの声で英語を話します
Exec=$ROOT/launch_app.sh
Icon=$ROOT/ui/assets/icon.svg
Terminal=false
Categories=AudioVideo;Audio;
StartupNotify=false
Actions=stop;

[Desktop Action stop]
Name=終了 (モデルも停止して GPU を解放)
Exec=$ROOT/stop_app.sh
EOF
chmod +x "$ENTRY"

STOP="$HOME/.local/share/applications/ikumeisai-stop.desktop"
cat > "$STOP" <<EOF
[Desktop Entry]
Type=Application
Name=文字起こし&翻訳AI を終了
Comment=アプリとモデルサーバーを終了して GPU メモリを解放します
Exec=$ROOT/stop_app.sh
Icon=process-stop
Terminal=false
Categories=AudioVideo;Audio;
EOF
chmod +x "$STOP"

DESKTOP=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
mkdir -p "$DESKTOP"
for f in "$ENTRY" "$STOP"; do
  cp "$f" "$DESKTOP/"
  d="$DESKTOP/$(basename "$f")"
  chmod +x "$d"
  # GNOME (Ubuntu) でダブルクリック起動を許可 ("Allow Launching" 相当)
  gio set "$d" metadata::trusted true 2>/dev/null || true
done
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
echo "登録しました: $DESKTOP/ikumeisai.desktop"

#!/bin/bash
# Mac のデスクトップに「文字起こし&翻訳AI.app」(起動) と「文字起こし&翻訳AI を終了.app」を作る。
#   bash mac/build_mac_app.sh [出力先 (既定: ~/Desktop)]
# EVO-X2 へ鍵認証で ssh できること (ssh sugakuka@192.168.1.12) が前提。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="${1:-$HOME/Desktop}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# アイコン: ui/assets/icon.svg → icns
qlmanage -t -s 1024 -o "$TMP" "$HERE/../ui/assets/icon.svg" >/dev/null 2>&1 || true
ICNS=""
if [[ -f "$TMP/icon.svg.png" ]]; then
  mkdir "$TMP/icon.iconset"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$TMP/icon.svg.png" --out "$TMP/icon.iconset/icon_${s}x${s}.png" >/dev/null
    sips -z $((s * 2)) $((s * 2)) "$TMP/icon.svg.png" --out "$TMP/icon.iconset/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$TMP/icon.iconset" -o "$TMP/icon.icns" && ICNS="$TMP/icon.icns"
fi

make_app() {  # name bundle-id args...
  local name=$1 id=$2; shift 2
  local app="$OUT/$name.app"
  local xname=${name//&/&amp;}  # plist (XML) 用
  rm -rf "$app"
  mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
  cp "$HERE/ikumeisai_mac.sh" "$app/Contents/Resources/ikumeisai_mac.sh"
  cat > "$app/Contents/MacOS/run" <<EOF
#!/bin/bash
exec "\$(dirname "\$0")/../Resources/ikumeisai_mac.sh" $* >> "\$HOME/Library/Logs/ikumeisai_mac.log" 2>&1
EOF
  chmod +x "$app/Contents/MacOS/run" "$app/Contents/Resources/ikumeisai_mac.sh"
  [[ -n "$ICNS" ]] && cp "$ICNS" "$app/Contents/Resources/icon.icns"
  cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$xname</string>
  <key>CFBundleDisplayName</key><string>$xname</string>
  <key>CFBundleIdentifier</key><string>$id</string>
  <key>CFBundleExecutable</key><string>run</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>LSUIElement</key><true/>
</dict></plist>
EOF
  touch "$app"
  echo "作成: $app"
}

make_app "文字起こし&翻訳AI" "jp.ac.meiji.ikumeisai.launch"
make_app "文字起こし&翻訳AI を終了" "jp.ac.meiji.ikumeisai.stop" stop

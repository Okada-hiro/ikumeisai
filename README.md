# ikumeisai — 日本語音声 → 英語音声 (声質クローン)

1. UI (`ui/`) でマイクを押して日本語で話す (最大 10 秒)。録音はブラウザで WAV にしてアプリサーバー (`server.py`, :7861) へ送り、`customer.wav` に保存
2. Qwen3-Omni-30B-A3B-Instruct (vLLM, :8000) が JSON で文字起こし・英訳。生成トークンごとに NDJSON で UI へ流し、文字起こし・英訳が 1 トークンずつ表示される → `transcription.txt` / `translation.txt`
3. Qwen3-TTS-12Hz-1.7B-Base (vLLM-Omni, :8001) が `ref_audio=customer.wav, ref_text=transcription, text=translation` で `translation.wav` を生成し、UI が再生

出力は `outputs/<日時>/` に保存されます。

## 起動 (ワンクリック)

アプリサーバーが TTS → Omni を起動し、テスト音声でウォームアップしてから UI を使える状態にします。その間 UI は「準備しています」画面で読み込み・ウォームアップの進み具合を表示します (モデル未起動なら数分、起動済みなら数秒)。

### EVO-X2 の画面で使う

一度だけ `./install_desktop.sh` を実行すると、デスクトップとアプリ一覧に「文字起こし&翻訳AI」が登録されます。クリックすると Firefox (専用プロファイル、マイク・自動再生は許可済み) が全画面で開きます。全画面は Alt+F4 で閉じます (モデルは動いたままなので、次の起動はすぐ)。GPU を空けたいときは「文字起こし&翻訳AI を終了」(`stop_app.sh`)。

### Mac の画面で使う

Mac で一度だけ `bash mac/build_mac_app.sh` を実行すると、デスクトップに「文字起こし&翻訳AI.app」と「文字起こし&翻訳AI を終了.app」ができます。起動すると EVO-X2 のアプリサーバーを画面なしで起動し (`launch_app.sh --server-only`)、SSH トンネル (Mac localhost:7861 → EVO-X2 127.0.0.1:7861) を張って Chrome のアプリウィンドウで UI を開きます。録音は Mac のマイク、音声は SSH で EVO-X2 に送られ、英語音声は Mac で再生されます。前提: Mac から `ssh sugakuka@192.168.1.12` が鍵認証で通ること (接続先は `IKUMEISAI_HOST` で変更可)。初回は macOS が Chrome のマイク使用許可を求めます。ログは `~/Library/Logs/ikumeisai_mac.log`。

### ターミナルから

```bash
cd ~/workspace/ikumeisai && ./start.sh   # UI は http://localhost:7861/
```

ログは `~/logs/{tts_server,vllm_server,app}.log`。アプリサーバーは 127.0.0.1 にだけ待ち受けます (別マシンからは SSH トンネル経由)。

`POST /api/process` (body: `audio/wav`) は NDJSON で `{"type": "transcript"|"translation", "text": <その時点の全文>, "done": bool}`、`status`、`audio` (`/api/audio/<session>/translation.wav`)、`timing` を返します。UI 側は `ui/assets/api-processor.js` (デモの `demo-processor.js` を置き換えたもの)。

## 動作確認

```bash
source ~/venvs/app/bin/activate && python test_samples.py
```

`test_audio/` の FLEURS ja_jp dev 3 本 (各 10 秒) で全工程を実行し、正解文と比較します。

## 環境 (ネイティブ Ubuntu 24.04 / Radeon 8060S gfx1151)

| venv | 用途 |
|---|---|
| `~/venvs/vllm` | vLLM 0.31.0+rocm723 (Qwen3-Omni) |
| `~/venvs/vllm-omni` | vLLM 0.31.0 + vLLM-Omni 0.31.0rc1 (Qwen3-TTS) |
| `~/venvs/app` | アプリサーバー (FastAPI, `server.py`) |

- セットアップ時に一度: `wsl_compat/apply_rocm_tts_patches.sh` (vLLM-Omni に TTS 高速化パッチ)、`wsl_compat/install_vllm_plugin.sh` (Omni の int8 プラグイン登録)。Omni は `wsl_compat/make_omni_w8a16g.py` で作る W8A16 モデルを使う
- ROCm 7.2.3 (`amdgpu-install --usecase=rocm --no-dkms`)。vLLM の torch 用に apt で `libopenmpi3t64` `libdw1t64` を入れる
- BIOS で iGPU (UMA) を 2G にし、カーネル引数 `ttm.pages_limit=28835840 ttm.page_pool_size=28835840` で GPU が使えるメモリ (GTT) を約 110GB にする
- `moe_configs/` — gfx1151 向けに調整した fused MoE の設定 (`serve_vllm.sh` が `VLLM_TUNED_CONFIG_FOLDER` で読み込む)。既定の設定だとデコードが 14.5 tok/s、M=1,2 を調整して 26.2 tok/s

`wsl_compat/` は WSL2 + librocdxg 時代の回避策 (ネイティブでは不要)。`wsl_compat/tune_moe.py` は MoE 調整用にネイティブでも使う (出力先は venv 内なので、生成後に `moe_configs/` へコピーする)。

## テスト音声の出典

`test_audio/sample*.wav` は [google/fleurs](https://huggingface.co/datasets/google/fleurs) (ja_jp, dev) の音声です (CC BY 4.0)。元ファイル名と正解文は `test_audio/samples.json` にあります。

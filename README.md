# ikumeisai — 日本語音声 → 英語音声 (声質クローン)

1. UI で「日本語で話してください」と表示し、マイク音声を `customer.wav` に保存
2. Qwen3-Omni-30B-A3B-Instruct (vLLM, :8000) が JSON で文字起こし・英訳 → `transcription.txt` / `translation.txt`
3. Qwen3-TTS-12Hz-1.7B-Base (vLLM-Omni, :8001) が `ref_audio=customer.wav, ref_text=transcription, text=translation` で `translation.wav` を生成
4. UI (:7861) に文字起こし・英訳を表示し、`translation.wav` を再生

出力は `outputs/<日時>/` に保存されます。

## 起動

```bash
cd ~/workspace/ikumeisai && ./start.sh
```

TTS → Omni → UI の順に起動します (初回は合計 5 分ほど)。ログは `~/logs/{tts_server,vllm_server,app}.log`。

## UI へのアクセス (Mac から)

ブラウザのマイクは `localhost` か HTTPS でしか使えないため、SSH トンネル経由で開きます。

```bash
ssh -p 2222 -N -L 7861:127.0.0.1:7861 sugakuka@192.168.1.12
```

その後 http://localhost:7861 を開く。

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
| `~/venvs/app` | Gradio UI |

- ROCm 7.2.3 (`amdgpu-install --usecase=rocm --no-dkms`)。vLLM の torch 用に apt で `libopenmpi3t64` `libdw1t64` を入れる
- BIOS で iGPU (UMA) を 2G にし、カーネル引数 `ttm.pages_limit=28835840 ttm.page_pool_size=28835840` で GPU が使えるメモリ (GTT) を約 110GB にする
- `moe_configs/` — gfx1151 向けに調整した fused MoE の設定 (`serve_vllm.sh` が `VLLM_TUNED_CONFIG_FOLDER` で読み込む)。既定の設定だとデコードが 14.5 tok/s、M=1,2 を調整して 26.2 tok/s

`wsl_compat/` は WSL2 + librocdxg 時代の回避策 (ネイティブでは不要)。`wsl_compat/tune_moe.py` は MoE 調整用にネイティブでも使う (出力先は venv 内なので、生成後に `moe_configs/` へコピーする)。

## テスト音声の出典

`test_audio/sample*.wav` は [google/fleurs](https://huggingface.co/datasets/google/fleurs) (ja_jp, dev) の音声です (CC BY 4.0)。元ファイル名と正解文は `test_audio/samples.json` にあります。

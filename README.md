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

## 環境 (WSL2 + librocdxg / Radeon 8060S gfx1151)

| venv | 用途 |
|---|---|
| `~/venvs/vllm` | vLLM 0.31.0+rocm723 (Qwen3-Omni) |
| `~/venvs/vllm-omni` | vLLM 0.31.0 + vLLM-Omni 0.31.0rc1 (Qwen3-TTS) |
| `~/venvs/app` | Gradio UI (torch 2.11.0+rocm7.13.0) |

CUDA は使えない (AMD GPU) ため、ROCm 版 vLLM で動かしています。WSL 向けの回避策は `wsl_compat/` にまとめてあります:

- `env.sh` — `HSA_ENABLE_DXG_DETECTION=1`、`ROCPROFILER_REGISTER_ENABLED=0` (rocprofiler が `/sys/class/kfd` 不在で異常終了するため)、`~/.local/lib/vllm-compat` (`libdw.so.1` と OpenMPI 4 C++ バインディングのスタブ `libmpi_cxx.so.40`、ソースは `mpi_cxx_stub.c`)
- `runtime/sitecustomize.py` — MIOpen の畳み込みを無効化 (新しい入力形状ごとに約 29 秒のカーネル探索が走り、vLLM の起動プロファイルが終わらないため)
- `patch_vllm_rocm.py` / `patch_vllm_omni_rocm.py` — amdsmi (ドライバ不在で失敗) の代わりに torch で GPU 情報を取得。venv を作り直したら再実行

## テスト音声の出典

`test_audio/sample*.wav` は [google/fleurs](https://huggingface.co/datasets/google/fleurs) (ja_jp, dev) の音声です (CC BY 4.0)。元ファイル名と正解文は `test_audio/samples.json` にあります。

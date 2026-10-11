import time

import gradio as gr

import config
import pipeline


def process(mic):
    if mic is None:
        raise gr.Error("音声が録音されていません。マイクボタンを押して日本語で話してください。")
    sr, audio = mic
    session = pipeline.new_session_dir()
    wav = pipeline.save_customer_wav(sr, audio, session)

    t0 = time.time()
    try:
        ja, en, _ = pipeline.transcribe_translate(wav, session)
    except Exception as e:
        raise gr.Error(f"文字起こし・翻訳に失敗しました: {e}")
    t1 = time.time()
    out_wav = pipeline.synthesize(wav, ja, en, session)
    t2 = time.time()

    status = f"保存先: {session.name}  |  文字起こし+英訳 {t1 - t0:.1f}s / 音声合成 {t2 - t1:.1f}s"
    # 録音欄は空に戻して次の録音を受け付ける (録音した音声は customer_audio で再生できる)
    return None, str(wav), ja, en, str(out_wav), status


def reset():
    return None, None, "", "", None, ""


with gr.Blocks(title="ikumeisai 音声翻訳") as demo:
    gr.Markdown("# 🎙️ 日本語で話してください\nマイクボタンを押して話し、終わったら停止して「翻訳する」を押してください。")
    mic = gr.Audio(sources=["microphone"], type="numpy", label="録音")
    with gr.Row():
        btn = gr.Button("翻訳する", variant="primary", scale=3)
        clear_btn = gr.Button("次のお客様", scale=1)
    customer_audio = gr.Audio(label="お客様の音声 (customer.wav)", type="filepath", interactive=False)
    with gr.Row():
        ja_box = gr.Textbox(label="文字起こし (transcription.txt)", lines=4)
        en_box = gr.Textbox(label="英訳 (translation.txt)", lines=4)
    out_audio = gr.Audio(label="英語音声 (translation.wav)", type="filepath", autoplay=True, interactive=False)
    status = gr.Markdown()

    outputs = [mic, customer_audio, ja_box, en_box, out_audio, status]
    btn.click(process, inputs=mic, outputs=outputs)
    clear_btn.click(reset, outputs=outputs)


if __name__ == "__main__":
    demo.queue().launch(server_name=config.APP_HOST, server_port=config.APP_PORT)

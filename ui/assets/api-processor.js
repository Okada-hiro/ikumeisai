/** Real backend adapter (replaces demo-processor.js). Same async generator interface. */

// MediaRecorder gives webm/ogg; the server has no ffmpeg, so decode here and send mono 16-bit WAV.
async function toWav(blob) {
  const ctx = new AudioContext();
  try {
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    const n = buf.length, ch = buf.numberOfChannels, sr = buf.sampleRate;
    const mono = new Float32Array(n);
    for (let c = 0; c < ch; c++) { const d = buf.getChannelData(c); for (let i = 0; i < n; i++) mono[i] += d[i] / ch; }
    const out = new DataView(new ArrayBuffer(44 + n * 2));
    const str = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
    str(0, 'RIFF'); out.setUint32(4, 36 + n * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
    out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
    out.setUint32(24, sr, true); out.setUint32(28, sr * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
    str(36, 'data'); out.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) { const s = Math.max(-1, Math.min(1, mono[i])); out.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
    return new Blob([out], { type: 'audio/wav' });
  } finally { ctx.close(); }
}

export async function* apiProcessor({ audioBlob, lang = 'en' }) {
  if (!audioBlob?.size) throw new Error('録音した音声が空です');
  const wav = await toWav(audioBlob);
  const res = await fetch(`/api/process?lang=${encodeURIComponent(lang)}`, { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav });
  if (!res.ok) {
    let msg = `サーバーエラー (${res.status})`;
    try { msg = (await res.json()).detail || msg; } catch { /* keep default */ }
    throw new Error(msg);
  }
  // NDJSON: one event per line, flushed per generated token — yield each as soon as it arrives.
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (value) buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
      if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.type === 'error') throw new Error(ev.message);
      if (ev.type === 'timing') { console.info('timing', ev); continue; }
      yield ev;
    }
    if (done) break;
  }
}

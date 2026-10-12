/** Gapless playback of streamed 16-bit mono PCM chunks with the Web Audio API. */
export class PcmStreamPlayer {
  /** @param {AudioContext} ctx  @param {number} lead seconds of headroom when (re)starting, to absorb jitter */
  constructor(ctx, lead = 0.15) {
    this.ctx = ctx; this.lead = lead; this.next = 0; this.sources = new Set();
    this.done = false; this.onDrained = null;
  }
  push(base64, sampleRate) {
    const bin = atob(base64), n = bin.length >> 1, f = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      if (v >= 0x8000) v -= 0x10000;
      f[i] = v / 0x8000;
    }
    const buf = this.ctx.createBuffer(1, n, sampleRate);
    buf.copyToChannel(f, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf; src.connect(this.ctx.destination);
    const now = this.ctx.currentTime;
    if (this.next < now + 0.01) this.next = now + this.lead; // first chunk, or the queue ran dry
    src.start(this.next);
    this.next += buf.duration;
    this.sources.add(src);
    src.onended = () => { this.sources.delete(src); if (this.done && !this.sources.size) this.onDrained?.(); };
  }
  /** Resolves when everything pushed so far has finished playing. */
  finish() {
    this.done = true;
    return new Promise(resolve => { if (!this.sources.size) resolve(); else this.onDrained = resolve; });
  }
  stop() {
    for (const s of this.sources) { try { s.stop(); } catch { /* already stopped */ } }
    this.sources.clear(); this.onDrained?.(); this.onDrained = null;
  }
}

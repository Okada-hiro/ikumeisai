/** Startup screen: polls /api/status until the model servers are loaded and warmed up. */
const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export function waitUntilReady() {
  const overlay = document.getElementById('bootOverlay');
  const list = document.getElementById('bootSteps');
  const elapsed = document.getElementById('bootElapsed');
  const note = document.getElementById('bootNote');
  return new Promise(resolve => {
    const tick = async () => {
      let st;
      try {
        st = await (await fetch('/api/status', { cache: 'no-store' })).json();
      } catch {
        note.textContent = 'アプリサーバーに接続しています…';
        setTimeout(tick, 1000);
        return;
      }
      list.innerHTML = '';
      for (const s of st.steps) {
        const li = document.createElement('li');
        li.className = `boot-step is-${s.status}`;
        const mark = document.createElement('span');
        mark.className = s.status === 'active' ? 'spinner' : 'boot-mark';
        mark.textContent = s.status === 'done' ? '✓' : s.status === 'error' ? '!' : '';
        const label = document.createElement('span');
        label.className = 'boot-label';
        label.textContent = s.label + (s.detail ? ` (${s.detail})` : '');
        const time = document.createElement('span');
        time.className = 'boot-time';
        time.textContent = s.elapsed != null ? fmt(s.elapsed) : '';
        li.append(mark, label, time);
        list.append(li);
      }
      elapsed.textContent = fmt(st.elapsed);
      if (st.phase === 'error') {
        note.textContent = '起動に失敗しました: ' + st.error;
        note.classList.add('is-error');
        return; // stop polling; relaunch after fixing
      }
      note.textContent = st.phase === 'warming' ? 'テスト音声で一度通して、すぐ使える状態にしています' : 'モデルを GPU に読み込んでいます (数分かかります)';
      if (st.phase === 'ready') {
        overlay.classList.add('is-done');
        setTimeout(() => overlay.remove(), 600);
        resolve();
        return;
      }
      setTimeout(tick, 1000);
    };
    tick();
  });
}

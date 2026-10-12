/** Reusable UI and recording orchestration. No hardcoded transcript or API endpoint. */
export class TranslationUI {
  constructor({ processor, maxRecordingMs = 10000 }) {
    if (typeof processor !== 'function') throw new TypeError('processor must be an async generator function');
    this.processor = processor;
    this.maxRecordingMs = maxRecordingMs;
    this.el = Object.fromEntries(['recordButton','recordStatus','statusDot','timer','jaText','enText','jaSpinner','enSpinner','jaState','enState','voiceStatus','voicePanel','replayButton'].map(id => [id,document.getElementById(id)]));
    this.shell = document.querySelector('.shell');
    this.mediaRecorder = null; this.mediaStream = null; this.chunks = []; this.recordStartedAt = 0;
    this.ticker = null; this.autoStop = null; this.busy = false; this.recording = false;
    this.audio = null; this.speech = null; this.lastAudioUrl = null; this.lastEnglish = '';
    this.el.recordButton.addEventListener('click', () => this.toggle());
    this.el.replayButton.addEventListener('click', () => this.playResult(this.lastAudioUrl, this.lastEnglish));
    window.addEventListener('pagehide', () => this.cleanup());
  }
  async toggle(){ if(this.recording){this.stopRecording();return;} if(this.busy)return; await this.startRecording(); }
  async startRecording(){
    this.stopPlayback();this.reset();
    try {
      if(!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error('このブラウザではマイク録音に対応していません。localhost または HTTPS で開いてください。');
      this.mediaStream = await navigator.mediaDevices.getUserMedia({audio:true});
      this.chunks=[];
      const rec = new MediaRecorder(this.mediaStream);
      this.mediaRecorder=rec;
      rec.ondataavailable=e=>{if(e.data.size)this.chunks.push(e.data);};
      rec.onstop=()=>this.finishRecording(rec.mimeType);
      rec.start(200);
      this.recordStartedAt=performance.now(); this.recording=true; this.shell.classList.add('is-recording');
      this.el.recordButton.setAttribute('aria-label','録音を停止');this.el.recordButton.setAttribute('aria-pressed','true');
      this.el.recordStatus.textContent='録音中 · もう一度押して停止';this.el.timer.textContent='00:00';
      this.ticker=setInterval(()=>{let s=Math.floor((performance.now()-this.recordStartedAt)/1000);this.el.timer.textContent=`00:${String(s).padStart(2,'0')}`;},150);
      if(this.maxRecordingMs>0)this.autoStop=setTimeout(()=>this.stopRecording(),this.maxRecordingMs);
    } catch(e) {this.releaseMic();this.el.recordStatus.textContent=e.name==='NotAllowedError'?'マイクのアクセスを許可してください':e.message;}
  }
  stopRecording(){
    if(!this.recording)return;
    this.recording=false;this.shell.classList.remove('is-recording');this.el.recordButton.setAttribute('aria-label','録音を開始');this.el.recordButton.setAttribute('aria-pressed','false');
    clearInterval(this.ticker);clearTimeout(this.autoStop);this.el.timer.textContent='';
    if(this.mediaRecorder?.state!=='inactive')this.mediaRecorder.stop();
    this.el.recordStatus.textContent='音声を処理しています';
  }
  async finishRecording(mimeType){
    const blob=new Blob(this.chunks,{type:mimeType||'audio/webm'});
    this.releaseMic();this.busy=true;this.el.recordButton.disabled=true;
    if(!blob.size){this.el.recordStatus.textContent='録音データがありません。もう一度お試しください';this.unlock();return;}
    this.setLoading('ja',true);this.setLoading('en',true);
    try {
      // processor yields {type:'transcript'|'translation'|'audio'|'status', ...}
      for await(const event of this.processor({audioBlob:blob,mimeType:blob.type})){
        if(event.type==='transcript')this.updateText('ja',event.text,event.done);
        if(event.type==='translation'){this.lastEnglish=event.text;this.updateText('en',event.text,event.done);}
        if(event.type==='status'&&event.message)this.el.voiceStatus.textContent=event.message;
        if(event.type==='audio'){
          this.lastAudioUrl=event.url||null;
          this.el.voiceStatus.textContent='あなたの声で英語音声を再生中';
          await this.playResult(this.lastAudioUrl,this.lastEnglish);
        }
      }
      this.setLoading('ja',false);this.setLoading('en',false);this.el.recordStatus.textContent='完了 · マイクを押して再録音';
    }catch(err){console.error(err);this.el.recordStatus.textContent='処理に失敗しました。もう一度お試しください';this.el.voiceStatus.textContent=err.message||'エラーが発生しました';this.setLoading('ja',false);this.setLoading('en',false);}finally{this.unlock();}
  }
  updateText(lang,text,done=false){let el=this.el[lang+'Text'];el.textContent=text;el.classList.toggle('streaming',!done);this.el[lang+'State'].textContent=done?'COMPLETE':'GENERATING';this.el[lang+'State'].classList.toggle('is-done',done);this.setLoading(lang,!done);}
  setLoading(lang,on){this.el[lang+'Spinner'].hidden=!on;}
  reset(){for(const lang of ['ja','en']){this.el[lang+'Text'].innerHTML=`<span class="placeholder">${lang==='ja'?'録音すると、ここに日本語の文字起こしが表示されます。':'英語の翻訳文がここに表示されます。'}</span>`;this.el[lang+'Text'].classList.remove('streaming');this.el[lang+'State'].textContent='READY';this.el[lang+'State'].classList.remove('is-done');this.setLoading(lang,false);}this.el.voicePanel.classList.remove('is-playing');this.el.voiceStatus.textContent='音声の生成を待っています';this.el.replayButton.hidden=true;this.lastEnglish='';this.lastAudioUrl=null;}
  async playResult(url,text){
    this.stopPlayback();this.el.replayButton.hidden=false;this.el.voicePanel.classList.add('is-playing');this.el.voiceStatus.textContent='あなたの声で英語音声を再生中';
    try{
      if(url){await new Promise((resolve,reject)=>{const audio=this.audio=new Audio(url);audio.onended=resolve;audio.onerror=()=>reject(new Error('音声を再生できませんでした'));audio.play().catch(reject);});}
      else if(window.speechSynthesis&&text){ // Demo fallback only: NOT voice cloning.
        await new Promise((resolve,reject)=>{const utterance=this.speech=new SpeechSynthesisUtterance(text);utterance.lang='en-US';utterance.rate=.93;utterance.onend=resolve;utterance.onerror=e=>e.error==='canceled'?resolve():reject(new Error('読み上げに失敗しました'));window.speechSynthesis.speak(utterance);});
      }else throw new Error('再生できる音声がありません');
      this.el.voiceStatus.textContent='英語音声の再生が完了しました';
    }catch(e){this.el.voiceStatus.textContent=e.message;}finally{this.el.voicePanel.classList.remove('is-playing');}
  }
  stopPlayback(){if(this.audio){this.audio.pause();this.audio=null;}if(window.speechSynthesis)window.speechSynthesis.cancel();this.el.voicePanel.classList.remove('is-playing');}
  releaseMic(){this.mediaStream?.getTracks().forEach(track=>track.stop());this.mediaStream=null;this.mediaRecorder=null;}
  unlock(){this.busy=false;this.el.recordButton.disabled=false;}
  cleanup(){clearInterval(this.ticker);clearTimeout(this.autoStop);this.releaseMic();this.stopPlayback();}
}

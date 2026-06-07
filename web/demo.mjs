// Browser demo: text + audio -> forced alignment, entirely client-side.
// Supports English and Mandarin.
import * as ort from '/node_modules/onnxruntime-web/dist/ort.min.mjs';
import { G2p } from '/dist/g2p.js';
import { G2pM } from '/dist/g2pm.js';
import { PhonemizerEn } from '/dist/phonemize-en.js';
import { PhonemizerZh } from '/dist/phonemize-zh.js';
import { PhonemizerJa } from '/dist/phonemize-ja.js';
import { ForcedAligner, CtcForcedAligner } from '/dist/index.js';
import {
  loadG2pAssets, loadPhoneVocab, loadG2pmAssets, loadPhoneVocabZh, loadPhoneVocabJa, decodeToMono16k,
} from '/dist/assets-web.js';
import { createTokenizer } from '/node_modules/tokana/dist/browser.js';

ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1; // no cross-origin isolation needed

const MODELS = {
  en: '/models/en_w2v2_fc_10ms/model_quantized.onnx',
  zh: '/models/zh_w2v2_tiny_fc_10ms/model_quantized.onnx',
  ja: '/models/japanese-hubert-base-phoneme-ctc/model_quantized.onnx',
};
const SAMPLES = {
  en: { wav: '/sample/sample.wav', text: 'the quick brown fox jumps over the lazy dog' },
  zh: { wav: '/sample/zh_sample.wav', text: '快速的棕色狐狸' },
  ja: { wav: '/sample/ja_sample.wav', text: '音声認識のテストです' },
};

// --- status + progress UI (no-ops in a headless/control-less context) ---
const el = (id) => document.getElementById(id);
function setStatus(msg) { const s = el('status'); if (s) s.textContent = msg; }
function setProgress(frac) {
  const p = el('prog'); if (!p) return;
  if (frac == null) { p.hidden = true; p.removeAttribute('value'); return; } // null = hide
  p.hidden = false;
  if (frac < 0) p.removeAttribute('value');                                  // <0 = indeterminate
  else p.value = frac;
}
const mb = (n) => (n / 1e6).toFixed(0);

// fetch with a streamed progress bar (model files are large: ~40–123 MB)
async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label} ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  if (!res.body) { setStatus(`loading ${label}…`); return new Uint8Array(await res.arrayBuffer()); }
  const reader = res.body.getReader();
  const chunks = []; let received = 0;
  setProgress(total ? 0 : -1);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); received += value.length;
    setStatus(`downloading ${label}… ${mb(received)}${total ? '/' + mb(total) : ''} MB`);
    if (total) setProgress(received / total);
  }
  setProgress(null);
  const buf = new Uint8Array(received); let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf;
}

// Japanese dictionary choice (standard IPADIC vs the much larger NEologd).
const JA_DICTS = {
  ipadic: { dir: '/models/ipadic-dict', note: 'IPADIC, ~12 MB' },
  neologd: { dir: '/models/neologd-dict', note: 'NEologd, ~190 MB' },
};
const jaDict = () => (el('dict') && el('dict').value) || 'ipadic';

const aligners = {};
async function getAligner(lang) {
  const key = lang === 'ja' ? `ja:${jaDict()}` : lang;   // re-build when the dict changes
  if (!aligners[key]) {
    aligners[key] = (async () => {
      // Japanese uses a CTC model (tokana morphology + hubert phoneme-CTC).
      if (lang === 'ja') {
        const fmt = jaDict();
        const d = JA_DICTS[fmt];
        setStatus(`loading dictionary… (${d.note})`);
        const [tokenizer, vocab] = await Promise.all([
          createTokenizer({ format: fmt, dicPath: d.dir }),
          loadPhoneVocabJa('/assets/'),
        ]);
        const phonemizer = new PhonemizerJa(tokenizer, vocab);
        const bytes = await fetchWithProgress(MODELS.ja, 'model (ja)');
        setStatus('initializing inference session…');
        const session = await ort.InferenceSession.create(bytes);
        return new CtcForcedAligner({ session, ort, phonemizer });
      }
      setStatus('loading g2p assets…');
      let phonemizer;
      if (lang === 'zh') {
        const [assets, vocab] = await Promise.all([loadG2pmAssets('/assets/'), loadPhoneVocabZh('/assets/')]);
        phonemizer = new PhonemizerZh(new G2pM(assets), vocab);
      } else {
        const [assets, vocab] = await Promise.all([loadG2pAssets('/assets/'), loadPhoneVocab('/assets/')]);
        phonemizer = new PhonemizerEn(new G2p(assets), vocab);
      }
      const bytes = await fetchWithProgress(MODELS[lang], `model (${lang})`);
      setStatus('initializing inference session…');
      const session = await ort.InferenceSession.create(bytes);
      return new ForcedAligner({ session, ort, phonemizer });
    })();
  }
  return aligners[key];
}

async function alignUrl(text, wavUrl, lang = 'en') {
  const aligner = await getAligner(lang);
  const buf = await fetch(wavUrl).then((r) => r.arrayBuffer());
  const waveform = await decodeToMono16k(buf);
  return aligner.align(waveform, text);
}

// expose for the headless test
window.__alignUrl = alignUrl;
// preload English; tolerate a missing en model (e.g. a Japanese-only checkout)
// so the demo still works for the other languages.
window.__ready = getAligner('en').then(() => true).catch((e) => { console.warn('en preload skipped:', e.message); return false; });

// --- minimal UI wiring (if the page has the controls) ---
const $ = (id) => document.getElementById(id);

// trim leading/trailing near-silence (RMS based) — real recordings have dead air
// at the ends, which otherwise makes the first/last phones stretch to fill it.
// Returns the trimmed waveform plus the leading offset (seconds) to re-base times.
function trimSilence(wave, sr = 16000) {
  const win = Math.floor(0.02 * sr);            // 20 ms windows
  const n = Math.ceil(wave.length / win);
  const rms = new Float32Array(n);
  let peak = 0;
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (let j = i * win; j < Math.min((i + 1) * win, wave.length); j++) { s += wave[j] * wave[j]; c++; }
    rms[i] = Math.sqrt(s / Math.max(1, c));
    peak = Math.max(peak, rms[i]);
  }
  const thr = Math.max(peak * 0.05, 1e-4);
  let a = 0; while (a < n && rms[a] < thr) a++;
  let b = n - 1; while (b > a && rms[b] < thr) b--;
  if (a >= b) return { wave, offset: 0 };
  const margin = Math.floor(0.05 * sr);         // keep 50 ms padding around speech
  const start = Math.max(0, a * win - margin);
  const end = Math.min(wave.length, (b + 1) * win + margin);
  return { wave: wave.subarray(start, end), offset: start / sr };
}

// decode an audio ArrayBuffer and align it against the current transcript + lang
async function alignBuffer(buf) {
  const lang = $('lang') ? $('lang').value : 'en';
  try {
    const aligner = await getAligner(lang);          // shows dict/model DL progress on first use
    setStatus('decoding audio…');
    const full = await decodeToMono16k(buf);
    const { wave, offset } = trimSilence(full);
    const dur = (wave.length / 16000).toFixed(1);
    setStatus(`aligning… (${dur}s audio)`);
    const t0 = performance.now();
    const { phones, words } = await aligner.align(wave, $('text').value);
    const ms = (performance.now() - t0).toFixed(0);
    // re-base times onto the original recording so the timeline matches playback
    const shift = (segs) => segs.map(([s, e, l]) => [+(s + offset).toFixed(2), +(e + offset).toFixed(2), l]);
    renderTimeline(shift(words), shift(phones));
    setStatus(`done — ${words.length} words / ${phones.length} phones / ${dur}s audio / ${ms} ms`);
  } catch (e) { setStatus('error: ' + e.message); setProgress(null); throw e; }
}

if ($('lang')) {
  $('lang').onchange = () => {
    const lang = $('lang').value;
    $('text').value = SAMPLES[lang].text;
    if ($('dict')) $('dict').hidden = lang !== 'ja';   // dictionary picker is JA-only
  };
}

// Align: uploaded file, or the bundled sample if none chosen
if ($('run')) {
  $('run').onclick = async () => {
    const lang = $('lang') ? $('lang').value : 'en';
    const file = $('audio').files[0];
    const buf = file ? await file.arrayBuffer() : await fetch(SAMPLES[lang].wav).then((r) => r.arrayBuffer());
    await alignBuffer(buf);
  };
}

// Record from the microphone (localhost is a secure context, so getUserMedia works)
let mediaRecorder = null, chunks = [], micStream = null;
if ($('record')) {
  $('record').onclick = async () => {
    const btn = $('record');
    if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); return; }
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) { $('status').textContent = 'microphone error: ' + e.message; return; }
    chunks = [];
    mediaRecorder = new MediaRecorder(micStream);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      micStream.getTracks().forEach((t) => t.stop());
      btn.textContent = '🎙 Record';
      btn.classList.remove('recording');
      const blob = new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' });
      if ($('playback')) { $('playback').src = URL.createObjectURL(blob); $('playback').hidden = false; }
      await alignBuffer(await blob.arrayBuffer());
    };
    mediaRecorder.start();
    btn.textContent = '⏹ Stop';
    btn.classList.add('recording');
    $('status').textContent = 'recording… read the transcript aloud, then press Stop';
  };
}

function renderTimeline(words, phones) {
  const out = $('out');
  if (!out) return;
  const xmax = phones.at(-1)?.[1] || 1;
  const row = (segs, cls) => `<div class="tier">${segs.map(([s, e, l]) =>
    `<span class="seg ${cls}" style="left:${(s / xmax) * 100}%;width:${((e - s) / xmax) * 100}%" title="${s}-${e}">${l}</span>`).join('')}</div>`;
  out.innerHTML = row(words, 'w') + row(phones, 'p');
}

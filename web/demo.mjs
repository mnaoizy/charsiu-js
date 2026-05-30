// Browser demo: text + audio -> forced alignment, entirely client-side.
// Supports English and Mandarin.
import * as ort from '/node_modules/onnxruntime-web/dist/ort.min.mjs';
import { G2p } from '/dist/g2p.js';
import { G2pM } from '/dist/g2pm.js';
import { PhonemizerEn } from '/dist/phonemize-en.js';
import { PhonemizerZh } from '/dist/phonemize-zh.js';
import { ForcedAligner } from '/dist/index.js';
import {
  loadG2pAssets, loadPhoneVocab, loadG2pmAssets, loadPhoneVocabZh, decodeToMono16k,
} from '/dist/assets-web.js';

ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1; // no cross-origin isolation needed

const MODELS = {
  en: '/models/en_w2v2_fc_10ms/model_quantized.onnx',
  zh: '/models/zh_w2v2_tiny_fc_10ms/model_quantized.onnx',
};
const SAMPLES = {
  en: { wav: '/sample/sample.wav', text: 'the quick brown fox jumps over the lazy dog' },
  zh: { wav: '/sample/zh_sample.wav', text: '快速的棕色狐狸' },
};

const aligners = {};
async function getAligner(lang) {
  if (!aligners[lang]) {
    aligners[lang] = (async () => {
      let phonemizer;
      if (lang === 'zh') {
        const [assets, vocab] = await Promise.all([loadG2pmAssets('/assets/'), loadPhoneVocabZh('/assets/')]);
        phonemizer = new PhonemizerZh(new G2pM(assets), vocab);
      } else {
        const [assets, vocab] = await Promise.all([loadG2pAssets('/assets/'), loadPhoneVocab('/assets/')]);
        phonemizer = new PhonemizerEn(new G2p(assets), vocab);
      }
      const session = await ort.InferenceSession.create(MODELS[lang]);
      return new ForcedAligner({ session, ort, phonemizer });
    })();
  }
  return aligners[lang];
}

async function alignUrl(text, wavUrl, lang = 'en') {
  const aligner = await getAligner(lang);
  const buf = await fetch(wavUrl).then((r) => r.arrayBuffer());
  const waveform = await decodeToMono16k(buf);
  return aligner.align(waveform, text);
}

// expose for the headless test
window.__alignUrl = alignUrl;
window.__ready = getAligner('en').then(() => true);

// --- minimal UI wiring (if the page has the controls) ---
const $ = (id) => document.getElementById(id);

// decode an audio ArrayBuffer and align it against the current transcript + lang
async function alignBuffer(buf) {
  const status = $('status');
  const lang = $('lang') ? $('lang').value : 'en';
  status.textContent = 'aligning…';
  try {
    const waveform = await decodeToMono16k(buf);
    const aligner = await getAligner(lang);
    const { phones, words } = await aligner.align(waveform, $('text').value);
    renderTimeline(words, phones);
    status.textContent = `done — ${words.length} words, ${phones.length} phones`;
  } catch (e) { status.textContent = 'error: ' + e.message; throw e; }
}

if ($('lang')) {
  $('lang').onchange = () => { $('text').value = SAMPLES[$('lang').value].text; };
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

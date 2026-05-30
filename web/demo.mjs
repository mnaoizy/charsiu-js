// Browser demo: text + audio -> forced alignment, entirely client-side.
import * as ort from '/node_modules/onnxruntime-web/dist/ort.min.mjs';
import { G2p } from '/dist/g2p.js';
import { PhonemizerEn } from '/dist/phonemize-en.js';
import { ForcedAligner } from '/dist/index.js';
import { loadG2pAssets, loadPhoneVocab, decodeToMono16k } from '/dist/assets-web.js';

ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1; // no cross-origin isolation needed

const MODEL_URL = '/models/en_w2v2_fc_10ms/model_quantized.onnx';
let alignerPromise;

async function getAligner() {
  if (!alignerPromise) {
    alignerPromise = (async () => {
      const [g2pAssets, phoneVocab, session] = await Promise.all([
        loadG2pAssets('/assets/'),
        loadPhoneVocab('/assets/'),
        ort.InferenceSession.create(MODEL_URL),
      ]);
      const phonemizer = new PhonemizerEn(new G2p(g2pAssets), phoneVocab);
      return new ForcedAligner({ session, ort, phonemizer });
    })();
  }
  return alignerPromise;
}

async function alignUrl(text, wavUrl) {
  const aligner = await getAligner();
  const buf = await fetch(wavUrl).then((r) => r.arrayBuffer());
  const waveform = await decodeToMono16k(buf);
  return aligner.align(waveform, text);
}

// expose for the headless test
window.__alignUrl = alignUrl;
window.__ready = getAligner().then(() => true);

// --- minimal UI wiring (if the page has the controls) ---
const $ = (id) => document.getElementById(id);
if ($('run')) {
  $('run').onclick = async () => {
    const status = $('status');
    try {
      status.textContent = 'aligning…';
      const text = $('text').value;
      const file = $('audio').files[0];
      const buf = file ? await file.arrayBuffer() : await fetch('/sample/sample.wav').then((r) => r.arrayBuffer());
      const waveform = await decodeToMono16k(buf);
      const aligner = await getAligner();
      const { phones, words } = await aligner.align(waveform, text);
      renderTimeline(words, phones);
      status.textContent = `done — ${words.length} words, ${phones.length} phones`;
    } catch (e) { status.textContent = 'error: ' + e.message; throw e; }
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

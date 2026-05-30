// Browser asset loader (fetch-based) for the English g2p + aligner.
// baseUrl points to the directory containing cmudict.json, g2p_gru.bin, etc.
import type { G2pAssets } from './types.js';

export async function loadG2pAssets(baseUrl = './assets/'): Promise<G2pAssets> {
  const j = (f: string) => fetch(baseUrl + f).then((r) => r.json());
  const [cmudict, homographs, vocab, manifest, binBuf] = await Promise.all([
    j('cmudict.json'), j('homographs.json'), j('g2p_vocab.json'), j('g2p_gru.json'),
    fetch(baseUrl + 'g2p_gru.bin').then((r) => r.arrayBuffer()),
  ]);
  return { cmudict, homographs, vocab, gru: { manifest, data: new Float32Array(binBuf) } };
}

export function loadPhoneVocab(baseUrl = './assets/'): Promise<Record<string, number>> {
  return fetch(baseUrl + 'vocab_en.json').then((r) => r.json());
}

// Decode any browser-supported audio file to 16 kHz mono Float32 in [-1, 1].
export async function decodeToMono16k(arrayBuffer: ArrayBuffer): Promise<Float32Array> {
  const w = window as unknown as Record<string, unknown>;
  const Ctx = (w.OfflineAudioContext || w.webkitOfflineAudioContext) as typeof OfflineAudioContext;
  const AudioCtx = (w.AudioContext || w.webkitAudioContext) as typeof AudioContext;
  const tmp = new AudioCtx();
  const decoded = await tmp.decodeAudioData(arrayBuffer.slice(0));
  tmp.close();
  const targetRate = 16000;
  const frames = Math.ceil(decoded.duration * targetRate);
  const off = new Ctx(1, frames, targetRate);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

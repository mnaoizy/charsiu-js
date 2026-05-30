// Core forced-alignment pipeline, ported to JS from charsiu's Python
// (processors.audio_preprocess + Charsiu.align + utils.forced_align/seq2duration).
//
// Pure functions over an ONNX session, so the same code runs in Node
// (onnxruntime-node) and the browser (onnxruntime-web / transformers.js).
import type { Segment, AlignSpec, AlignSession, OrtLike } from './types.js';

// --- audio: zero-mean / unit-var normalization (wav2vec2 feature extractor) ---
export function normalize(waveform: Float32Array): Float32Array {
  let mean = 0;
  for (const x of waveform) mean += x;
  mean /= waveform.length;
  let varSum = 0;
  for (const x of waveform) varSum += (x - mean) * (x - mean);
  const std = Math.sqrt(varSum / waveform.length + 1e-7);
  const out = new Float32Array(waveform.length);
  for (let i = 0; i < waveform.length; i++) out[i] = (waveform[i] - mean) / std;
  return out;
}

// --- softmax over the last axis of a (frames x vocab) flat array ---
export function softmaxRows(logits: ArrayLike<number>, frames: number, vocab: number): Float32Array {
  const out = new Float32Array(frames * vocab);
  for (let f = 0; f < frames; f++) {
    const base = f * vocab;
    let max = -Infinity;
    for (let c = 0; c < vocab; c++) max = Math.max(max, logits[base + c]);
    let sum = 0;
    for (let c = 0; c < vocab; c++) { const e = Math.exp(logits[base + c] - max); out[base + c] = e; sum += e; }
    for (let c = 0; c < vocab; c++) out[base + c] /= sum;
  }
  return out;
}

// --- DTW forced alignment, equivalent to librosa.sequence.dtw with
//     step_sizes_sigma = [[1,1],[1,0]] over C = -cost[:, phoneIds].
//     Returns, for each input frame, the index into phoneIds it aligns to. ---
export function forcedAlign(cost: ArrayLike<number>, frames: number, vocab: number, phoneIds: number[]): Int32Array {
  const M = phoneIds.length;
  const C = (i: number, j: number) => -cost[i * vocab + phoneIds[j]]; // cost to minimize
  const D = new Float64Array(frames * M).fill(Infinity);
  const back = new Int8Array(frames * M);                 // 1=diag(i-1,j-1), 0=vert(i-1,j)

  D[0] = C(0, 0);                                         // frame0 must be phone0
  for (let i = 1; i < frames; i++) {
    const maxJ = Math.min(i, M - 1);                       // can't reach phone j before frame j
    for (let j = 0; j <= maxJ; j++) {
      const vert = D[(i - 1) * M + j];                     // stay on same phone
      const diag = j > 0 ? D[(i - 1) * M + (j - 1)] : Infinity;
      let prev: number, dir: number;
      if (diag <= vert) { prev = diag; dir = 1; } else { prev = vert; dir = 0; }
      if (prev === Infinity) continue;
      D[i * M + j] = C(i, j) + prev;
      back[i * M + j] = dir;
    }
  }

  // backtrack from (frames-1, M-1) to (0,0)
  const align = new Int32Array(frames);
  let j = M - 1;
  for (let i = frames - 1; i >= 0; i--) {
    align[i] = j;
    if (back[i * M + j] === 1) j -= 1;                     // diagonal -> previous phone
  }
  return align;
}

// --- group consecutive identical labels into [start, end, label] segments ---
export function seq2duration(phones: string[], resolution = 0.01): Segment[] {
  const out: Segment[] = [];
  let counter = 0;
  for (let i = 0; i < phones.length; ) {
    let j = i;
    while (j < phones.length && phones[j] === phones[i]) j++;
    const len = j - i;
    out.push([
      +(counter * resolution).toFixed(2),
      +((counter + len) * resolution).toFixed(2),
      phones[i],
    ]);
    counter += len;
    i = j;
  }
  return out;
}

// --- full pipeline: normalized waveform + phone spec -> aligned segments ---
export async function align(session: AlignSession, ort: OrtLike, waveform: Float32Array, spec: AlignSpec): Promise<Segment[]> {
  const norm = normalize(waveform);
  const feeds = { input_values: new ort.Tensor('float32', norm, [1, norm.length]) };
  const { logits } = await session.run(feeds);
  const [, frames, vocab] = logits.dims;
  const cost = softmaxRows(logits.data, frames, vocab);

  const { phone_ids, sil_idx, sil_threshold, resolution, id2phone } = spec;

  // argmax per frame
  const preds = new Int32Array(frames);
  for (let f = 0; f < frames; f++) {
    let best = 0, bestv = -Infinity;
    for (let c = 0; c < vocab; c++) { const v = cost[f * vocab + c]; if (v > bestv) { bestv = v; best = c; } }
    preds[f] = best;
  }

  // sil mask: short silence runs (< threshold) are demoted to -1 (treated as speech)
  const silMask = new Int32Array(frames);
  for (let i = 0; i < frames; ) {
    let j = i; while (j < frames && preds[j] === preds[i]) j++;
    const runLen = j - i;
    const demote = preds[i] === sil_idx && runLen < sil_threshold;
    for (let k = i; k < j; k++) silMask[k] = demote ? -1 : preds[i];
    i = j;
  }

  // non-silence frames -> DTW against the target phone sequence (without end sils)
  const nonsil: number[] = [];
  for (let f = 0; f < frames; f++) if (silMask[f] !== sil_idx) nonsil.push(f);
  const target = phone_ids.slice(1, -1);

  // build a compact cost over only non-sil frames
  const sub = new Float32Array(nonsil.length * vocab);
  for (let k = 0; k < nonsil.length; k++) sub.set(cost.subarray(nonsil[k] * vocab, nonsil[k] * vocab + vocab), k * vocab);
  const alignIdx = forcedAlign(sub, nonsil.length, vocab, target);
  // id2phone already holds the canonical label (stress-less for EN, tone-ful for ZH)
  const alignedPhones = Array.from(alignIdx, (p) => id2phone[String(target[p])]);

  // merge silence back into the full timeline
  const merged: string[] = [];
  let count = 0;
  for (let f = 0; f < frames; f++) {
    if (silMask[f] === sil_idx) merged.push('[SIL]');
    else merged.push(alignedPhones[count++]);
  }
  return seq2duration(merged, resolution);
}

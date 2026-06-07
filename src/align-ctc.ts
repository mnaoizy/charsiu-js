// CTC forced alignment, for acoustic models with a CTC head (a blank/PAD class
// dominates non-emitting frames) rather than charsiu's frame-classification head.
// Used by the Japanese aligner (prj-beatrice/japanese-hubert-base-phoneme-ctc):
// most frames argmax to blank, so the DTW in ./align.ts (which assumes every
// frame is a real phone or [SIL]) doesn't apply. This is the standard Viterbi
// over the blank-extended target sequence (cf. torchaudio.functional.forced_align).
//
// Pure functions over an ONNX session, so the same code runs in Node and the
// browser, mirroring ./align.ts.
import { normalize } from './align.js';
import type { Segment, AlignSession, OrtLike } from './types.js';

// --- log-softmax over the last axis of a (frames x vocab) flat array ---
export function logSoftmaxRows(logits: ArrayLike<number>, frames: number, vocab: number): Float64Array {
  const out = new Float64Array(frames * vocab);
  for (let f = 0; f < frames; f++) {
    const base = f * vocab;
    let max = -Infinity;
    for (let c = 0; c < vocab; c++) max = Math.max(max, logits[base + c]);
    let sum = 0;
    for (let c = 0; c < vocab; c++) sum += Math.exp(logits[base + c] - max);
    const logSum = Math.log(sum);
    for (let c = 0; c < vocab; c++) out[base + c] = logits[base + c] - max - logSum;
  }
  return out;
}

// --- CTC Viterbi forced alignment. logp is (frames x vocab) log-probs; targets
//     is the phone-id sequence (no blanks). Returns, for each frame, the index
//     into `targets` it belongs to, with -1 for leading silence (blank frames
//     before the first emission). Trailing/inter-phone blank frames carry forward
//     to the preceding phone. ---
export function ctcForcedAlign(logp: ArrayLike<number>, frames: number, vocab: number, targets: number[], blank: number): Int32Array {
  const L = targets.length;
  const frameTok = new Int32Array(frames).fill(-1);
  if (L === 0) return frameTok;

  // blank-extended sequence: [blank, t0, blank, t1, blank, ..., t_{L-1}, blank]
  const ext = new Int32Array(2 * L + 1);
  for (let i = 0; i < L; i++) { ext[2 * i] = blank; ext[2 * i + 1] = targets[i]; }
  ext[2 * L] = blank;
  const S = ext.length;
  const lp = (t: number, id: number) => logp[t * vocab + id];

  const NEG = -1e30;
  const a = new Float64Array(frames * S).fill(NEG);
  const bp = new Int32Array(frames * S);
  a[0] = lp(0, ext[0]);                                   // state 0 (blank)
  if (S > 1) a[1] = lp(0, ext[1]);                        // state 1 (first phone)

  for (let t = 1; t < frames; t++) {
    for (let s = 0; s < S; s++) {
      let best = a[(t - 1) * S + s], arg = s;             // stay
      if (s - 1 >= 0 && a[(t - 1) * S + (s - 1)] > best) { best = a[(t - 1) * S + (s - 1)]; arg = s - 1; }
      // skip the intervening blank only between two *different* phones
      if (s - 2 >= 0 && ext[s] !== blank && ext[s] !== ext[s - 2] && a[(t - 1) * S + (s - 2)] > best) {
        best = a[(t - 1) * S + (s - 2)]; arg = s - 2;
      }
      a[t * S + s] = best + lp(t, ext[s]);
      bp[t * S + s] = arg;
    }
  }

  // terminate at the final blank or the last phone, whichever is likelier
  let s = a[(frames - 1) * S + (S - 1)] >= a[(frames - 1) * S + (S - 2)] ? S - 1 : S - 2;
  const path = new Int32Array(frames);
  for (let t = frames - 1; t >= 0; t--) { path[t] = s; s = bp[t * S + s]; }

  // frames -> phone index, carrying the last emitted phone across blank frames
  let cur = -1;
  for (let t = 0; t < frames; t++) {
    if (ext[path[t]] !== blank) cur = (path[t] - 1) >> 1;
    frameTok[t] = cur;
  }
  return frameTok;
}

// --- group a per-frame phone-index array into [start, end, label] segments;
//     index -1 becomes the silence label. ---
export function ctcSegments(frameTok: ArrayLike<number>, targets: number[], id2phone: Record<string, string>,
                            resolution: number, silLabel = '[SIL]'): Segment[] {
  const frames = frameTok.length;
  const out: Segment[] = [];
  for (let i = 0; i < frames; ) {
    let j = i;
    while (j < frames && frameTok[j] === frameTok[i]) j++;
    const idx = frameTok[i];
    const label = idx < 0 ? silLabel : id2phone[String(targets[idx])];
    out.push([+(i * resolution).toFixed(2), +(j * resolution).toFixed(2), label]);
    i = j;
  }
  return out;
}

// --- word-level segments: re-map each phone-index to its word (via per-word
//     phone counts) and group contiguous frames of the same word. ---
export function ctcWordSegments(frameTok: ArrayLike<number>, groupLens: number[], words: string[], resolution: number): Segment[] {
  const tok2word: number[] = [];
  groupLens.forEach((n, w) => { for (let k = 0; k < n; k++) tok2word.push(w); });
  const wordOf = (ti: number) => (ti < 0 ? -1 : tok2word[ti]);
  const frames = frameTok.length;
  const out: Segment[] = [];
  for (let i = 0; i < frames; ) {
    const wi = wordOf(frameTok[i]);
    let j = i + 1;
    while (j < frames && wordOf(frameTok[j]) === wi) j++;
    if (wi >= 0) out.push([+(i * resolution).toFixed(2), +(j * resolution).toFixed(2), words[wi]]);
    i = j;
  }
  return out;
}

export interface CtcAlignSpec {
  /** Forced transcription as phone ids (no blanks). */
  target_ids: number[];
  /** id -> phone label. */
  id2phone: Record<string, string>;
  /** CTC blank/PAD id. */
  blank: number;
  /** Seconds per frame (e.g. 0.02 for a 20 ms-stride hubert-base). */
  resolution: number;
  /** Label for leading/trailing silence. Default '[SIL]'. */
  sil_label?: string;
  /** Apply wav2vec2 zero-mean/unit-var normalization. Default false (hubert-base). */
  do_normalize?: boolean;
}

// --- run the model and CTC-align: returns the per-frame phone index (frameTok)
//     used to build both phone and word segments. ---
export async function ctcFrameAlign(session: AlignSession, ort: OrtLike, waveform: Float32Array,
                                    targetIds: number[], blank: number, doNormalize = false): Promise<{ frameTok: Int32Array; frames: number }> {
  const wav = doNormalize ? normalize(waveform) : waveform;
  const feeds = { input_values: new ort.Tensor('float32', wav, [1, wav.length]) };
  const { logits } = await session.run(feeds);
  const [, frames, vocab] = logits.dims;
  const logp = logSoftmaxRows(logits.data, frames, vocab);
  return { frameTok: ctcForcedAlign(logp, frames, vocab, targetIds, blank), frames };
}

// --- full pipeline: waveform + CTC spec -> aligned phone segments ---
export async function alignCtc(session: AlignSession, ort: OrtLike, waveform: Float32Array, spec: CtcAlignSpec): Promise<Segment[]> {
  const { frameTok } = await ctcFrameAlign(session, ort, waveform, spec.target_ids, spec.blank, spec.do_normalize ?? false);
  return ctcSegments(frameTok, spec.target_ids, spec.id2phone, spec.resolution, spec.sil_label ?? '[SIL]');
}

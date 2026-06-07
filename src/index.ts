// charsiu-js public API (runtime-agnostic core).
// Bring your own ONNX session + ort binding (onnxruntime-node or -web), a G2p,
// and a Phonemizer. For Node, use createNodeAligner from ./aligner-node.ts.
import { align } from './align.js';
import type { Segment, PhonemizerLike, AlignSession, OrtLike } from './types.js';

export { G2p, normalizeNumbers } from './g2p.js';
export { PhonemizerEn } from './phonemize-en.js';
export { normalize, softmaxRows, forcedAlign, seq2duration, align } from './align.js';
export type { Segment, PhonemizerLike, CtcPhonemizerLike, AlignSpec, G2pAssets, G2pmAssets } from './types.js';

// Japanese / CTC building blocks (runtime-agnostic; browser users wire these
// with onnxruntime-web + tokana, Node users use createNodeAlignerJa).
export { kanaToPhonemes } from './g2p-ja.js';
export { PhonemizerJa } from './phonemize-ja.js';
export { logSoftmaxRows, ctcForcedAlign, ctcSegments, ctcWordSegments, ctcFrameAlign, alignCtc } from './align-ctc.js';
export { CtcForcedAligner } from './aligner-ctc.js';
export type { CtcAlignSpec } from './align-ctc.js';
export type { CtcForcedAlignerOptions, CtcAlignResult } from './aligner-ctc.js';

export interface AlignResult {
  /** Phone-level segments; `[SIL]` marks silence. */
  phones: Segment[];
  /** Word-level segments. */
  words: Segment[];
  /** The phone-id sequence (with leading/trailing silence) fed to alignment. */
  phoneIds: number[];
}

export interface ForcedAlignerOptions {
  session: AlignSession;
  ort: OrtLike;
  phonemizer: PhonemizerLike;
  /** Max silence run length (frames) demoted to speech. Default 4. */
  silThreshold?: number;
  /** Seconds per frame. Default 0.01. */
  resolution?: number;
}

/** Runtime-agnostic forced aligner. */
export class ForcedAligner {
  private session: AlignSession;
  private ort: OrtLike;
  private phonemizer: PhonemizerLike;
  private silThreshold: number;
  private resolution: number;

  constructor({ session, ort, phonemizer, silThreshold = 4, resolution = 0.01 }: ForcedAlignerOptions) {
    this.session = session;
    this.ort = ort;
    this.phonemizer = phonemizer;
    this.silThreshold = silThreshold;
    this.resolution = resolution;
  }

  // waveform: Float32Array of 16 kHz mono PCM in [-1, 1]; text: transcript.
  async align(waveform: Float32Array, text: string): Promise<AlignResult> {
    const { groups, words } = this.phonemizer.getPhonesAndWords(text);
    const phoneIds = this.phonemizer.getPhoneIds(groups);
    const spec = {
      phone_ids: phoneIds,
      sil_idx: this.phonemizer.silIdx,
      sil_threshold: this.silThreshold,
      resolution: this.resolution,
      id2phone: this.phonemizer.id2phone,
    };
    const phones = await align(this.session, this.ort, waveform, spec);
    const wordSegs = this.phonemizer.alignWords(phones, groups, words);
    return { phones, words: wordSegs, phoneIds };
  }
}

export interface TextGridTier { name: string; intervals: Segment[]; }

// Praat TextGrid (short format) for one or more interval tiers.
export function toTextGrid(tiers: TextGridTier[]): string {
  const xmax = Math.max(...tiers.map((t) => t.intervals.at(-1)?.[1] ?? 0));
  const L: string[] = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '',
    '0', String(xmax), '<exists>', String(tiers.length)];
  for (const { name, intervals } of tiers) {
    L.push('"IntervalTier"', `"${name}"`, '0', String(xmax), String(intervals.length));
    for (const [s, e, label] of intervals) L.push(String(s), String(e), `"${label}"`);
  }
  return L.join('\n') + '\n';
}

// Runtime-agnostic forced aligner for CTC acoustic models (e.g. the Japanese
// hubert phoneme-CTC model). Mirrors ForcedAligner in ./index.ts but uses CTC
// forced alignment (./align-ctc.ts) instead of charsiu's frame-classification
// DTW. Bring your own ONNX session + ort binding and a CtcPhonemizerLike.
import { ctcFrameAlign, ctcSegments, ctcWordSegments } from './align-ctc.js';
import type { Segment, AlignSession, OrtLike, CtcPhonemizerLike } from './types.js';

export interface CtcAlignResult {
  /** Phone-level segments; `[SIL]` marks silence. */
  phones: Segment[];
  /** Word-level segments. */
  words: Segment[];
  /** The phone-id target sequence fed to alignment. */
  phoneIds: number[];
}

export interface CtcForcedAlignerOptions {
  session: AlignSession;
  ort: OrtLike;
  phonemizer: CtcPhonemizerLike;
  /** Seconds per frame. Default 0.02 (20 ms-stride hubert/wav2vec2 base). */
  resolution?: number;
  /** Apply wav2vec2 zero-mean/unit-var normalization. Default false (hubert-base). */
  doNormalize?: boolean;
  /** Label for leading/trailing silence. Default '[SIL]'. */
  silLabel?: string;
}

/** Forced aligner for CTC models. */
export class CtcForcedAligner {
  private session: AlignSession;
  private ort: OrtLike;
  private phonemizer: CtcPhonemizerLike;
  private resolution: number;
  private doNormalize: boolean;
  private silLabel: string;

  constructor({ session, ort, phonemizer, resolution = 0.02, doNormalize = false, silLabel = '[SIL]' }: CtcForcedAlignerOptions) {
    this.session = session;
    this.ort = ort;
    this.phonemizer = phonemizer;
    this.resolution = resolution;
    this.doNormalize = doNormalize;
    this.silLabel = silLabel;
  }

  // waveform: Float32Array of 16 kHz mono PCM in [-1, 1]; text: transcript.
  async align(waveform: Float32Array, text: string): Promise<CtcAlignResult> {
    const { words, targetIds, groupLens } = this.phonemizer.phonemize(text);
    const { frameTok } = await ctcFrameAlign(this.session, this.ort, waveform, targetIds, this.phonemizer.blankIdx, this.doNormalize);
    const phones = ctcSegments(frameTok, targetIds, this.phonemizer.id2phone, this.resolution, this.silLabel);
    const wordSegs = ctcWordSegments(frameTok, groupLens, words, this.resolution);
    return { phones, words: wordSegs, phoneIds: targetIds };
  }
}

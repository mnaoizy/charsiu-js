// Shared types for charsiu-js.

/** A time-aligned segment: [startSeconds, endSeconds, label]. */
export type Segment = [start: number, end: number, label: string];

/** A flat weight tensor: row-major `data` with its `shape`. */
export interface Weight {
  data: Float32Array;
  shape: number[];
}

/** A blob of named weight tensors plus the manifest describing their layout. */
export interface WeightBlob {
  manifest: Record<string, { offset: number; shape: number[] }>;
  data: Float32Array;
}

export interface G2pAssets {
  /** word -> space-joined ARPABET (first pronunciation). */
  cmudict: Record<string, string>;
  /** word -> [pron1, pron2, pos1]. */
  homographs: Record<string, [string[], string[], string]>;
  vocab: { graphemes: string[]; phonemes: string[] };
  gru: WeightBlob;
}

export interface G2pmAssets {
  /** hanzi -> list of pinyin (with tone digits). */
  cedict: Record<string, string[]>;
  char2idx: Record<string, number>;
  /** class id -> pinyin. */
  idx2class: string[];
  lstm: WeightBlob;
}

/** Turns text into a CTC target sequence (phone ids, no blanks) plus the
 *  per-word phone grouping needed to recover word-level segments. */
export interface CtcPhonemizerLike {
  id2phone: Record<string, string>;
  /** CTC blank/PAD id. */
  blankIdx: number;
  /** Silence id (or -1 if none). */
  silIdx: number;
  phonemize(text: string): { groups: string[][]; words: string[]; targetIds: number[]; groupLens: number[] };
}

/** Turns text into phone ids and maps phone segments back to words. */
export interface PhonemizerLike {
  silIdx: number;
  id2phone: Record<string, string>;
  getPhonesAndWords(text: string): { groups: string[][]; words: string[] };
  getPhoneIds(groups: string[][]): number[];
  alignWords(preds: Segment[], groups: string[][], words: string[]): Segment[];
}

export interface AlignSpec {
  phone_ids: number[];
  sil_idx: number;
  sil_threshold: number;
  resolution: number;
  id2phone: Record<string, string>;
}

/** Minimal shape of an onnxruntime tensor we read back. */
export interface OrtValue {
  data: ArrayLike<number>;
  dims: readonly number[];
}

/** Minimal shape of the onnxruntime module/namespace we use. */
export interface OrtLike {
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown;
}

/** Minimal shape of an onnxruntime InferenceSession. */
export interface AlignSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtValue>>;
}

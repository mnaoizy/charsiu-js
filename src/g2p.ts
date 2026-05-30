// English grapheme-to-phoneme, ported from g2p_en (Park & Kim).
// Pipeline: number/text normalization -> tokenize -> per-word
//   punctuation -> homograph -> CMUdict -> OOV GRU seq2seq.
// Output: ARPABET phones (with stress digits) and " " word separators,
// matching g2p_en(...) exactly for CMUdict/OOV words.
//
// NOTE: g2p_en uses an nltk POS tagger only to disambiguate ~371 homographs.
// We don't bundle a POS tagger yet, so homographs fall back to their default
// (second) pronunciation. Everything else is bit-faithful to the Python.
import type { G2pAssets, Weight } from './types.js';

// ---------- number normalization (port of g2p_en/expand.py) ----------
const ONES = ['zero','one','two','three','four','five','six','seven','eight','nine',
  'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen'];
const TENS = ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'];
const SCALES = ['','thousand','million','billion','trillion','quadrillion'];

function threeDigits(n: number): string[] { // 0..999 -> words (no "and")
  const out: string[] = [];
  if (n >= 100) { out.push(ONES[Math.floor(n / 100)], 'hundred'); n %= 100; }
  if (n >= 20) { const t = TENS[Math.floor(n / 10)]; out.push(n % 10 ? `${t}-${ONES[n % 10]}` : t); }
  else if (n > 0) out.push(ONES[n]);
  return out;
}
function cardinal(n: number): string { // non-negative integer -> words, inflect andword='' style
  if (n === 0) return 'zero';
  const groups: number[] = [];
  while (n > 0) { groups.push(n % 1000); n = Math.floor(n / 1000); }
  const parts: string[] = [];
  for (let i = groups.length - 1; i >= 0; i--) {
    if (groups[i] === 0) continue;
    parts.push(...threeDigits(groups[i]));
    if (i > 0) parts.push(SCALES[i]);
  }
  return parts.join(' ');
}
function twoDigitGroups(num: number, zeroWord: string): string { // inflect group=2 reading (years)
  const s = String(num);
  const padded = s.length % 2 ? '0' + s : s;
  const chunks: string[] = [];
  for (let i = 0; i < padded.length; i += 2) {
    const c = parseInt(padded.slice(i, i + 2), 10);
    if (c === 0) chunks.push(zeroWord, zeroWord);
    else if (c < 10) chunks.push(zeroWord, ONES[c]);
    else chunks.push(threeDigits(c).join(' '));
  }
  return chunks.join(' ');
}
function expandNumber(numStr: string): string {
  const num = parseInt(numStr, 10);
  if (num > 1000 && num < 3000) {
    if (num === 2000) return 'two thousand';
    if (num > 2000 && num < 2010) return 'two thousand ' + cardinal(num % 100);
    if (num % 100 === 0) return cardinal(Math.floor(num / 100)) + ' hundred';
    return twoDigitGroups(num, 'oh');
  }
  return cardinal(num);
}
function expandDollars(m: string): string {
  const parts = m.split('.');
  if (parts.length > 2) return m + ' dollars';
  const dollars = parts[0] ? parseInt(parts[0], 10) : 0;
  const cents = parts.length > 1 && parts[1] ? parseInt(parts[1], 10) : 0;
  if (dollars && cents) return `${dollars} ${dollars === 1 ? 'dollar' : 'dollars'}, ${cents} ${cents === 1 ? 'cent' : 'cents'}`;
  if (dollars) return `${dollars} ${dollars === 1 ? 'dollar' : 'dollars'}`;
  if (cents) return `${cents} ${cents === 1 ? 'cent' : 'cents'}`;
  return 'zero dollars';
}
export function normalizeNumbers(text: string): string {
  text = text.replace(/([0-9][0-9,]+[0-9])/g, (m) => m.replace(/,/g, ''));
  text = text.replace(/£([0-9,]*[0-9]+)/g, (_, n) => `${n} pounds`);
  text = text.replace(/\$([0-9.,]*[0-9]+)/g, (_, n) => expandDollars(n));
  text = text.replace(/([0-9]+\.[0-9]+)/g, (m) => m.replace('.', ' point '));
  text = text.replace(/[0-9]+(st|nd|rd|th)/g, (m) => ordinal(m));
  text = text.replace(/[0-9]+/g, (m) => expandNumber(m));
  return text;
}
function ordinal(m: string): string {
  const n = parseInt(m, 10);
  const words = cardinal(n).split(' ');
  const last = words.pop() as string;
  const ORD: Record<string, string> = { one: 'first', two: 'second', three: 'third', five: 'fifth',
    eight: 'eighth', nine: 'ninth', twelve: 'twelfth' };
  let o = ORD[last] || (last.endsWith('y') ? last.slice(0, -1) + 'ieth' : last + 'th');
  if (last.includes('-')) { const [a, b] = last.split('-'); o = a + '-' + (ORD[b] || b + 'th'); }
  return [...words, o].join(' ');
}

// ---------- GRU seq2seq for OOV words (port of g2p_en predict) ----------
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export class G2p {
  private cmu: Record<string, string>;
  private homographs: Record<string, [string[], string[], string]>;
  private graphemes: string[];
  private phonemes: string[];
  private g2idx: Record<string, number>;
  private idx2p: string[];
  private w: Record<string, Weight> = {};
  private H: number;
  private E: number;

  constructor(assets: G2pAssets) {
    this.cmu = assets.cmudict;
    this.homographs = assets.homographs;
    this.graphemes = assets.vocab.graphemes;
    this.phonemes = assets.vocab.phonemes;
    this.g2idx = Object.fromEntries(this.graphemes.map((g, i) => [g, i]));
    this.idx2p = this.phonemes;
    const { manifest, data } = assets.gru;
    for (const [k, { offset, shape }] of Object.entries(manifest)) {
      const len = shape.reduce((a, b) => a * b, 1);
      this.w[k] = { data: data.subarray(offset, offset + len), shape };
    }
    this.H = this.w.enc_w_hh.shape[1]; // hidden size
    this.E = this.w.enc_emb.shape[1];  // embedding size
  }

  // single GRU cell; x,h are length E/H. Returns new h (length H).
  private grucell(x: ArrayLike<number>, h: ArrayLike<number>, w_ih: Weight, w_hh: Weight, b_ih: Weight, b_hh: Weight): Float64Array {
    const H = this.H, gates = 3 * H;
    const ih = new Float64Array(gates), hh = new Float64Array(gates);
    for (let o = 0; o < gates; o++) {
      let s = b_ih.data[o]; const rowI = o * w_ih.shape[1];
      for (let i = 0; i < w_ih.shape[1]; i++) s += x[i] * w_ih.data[rowI + i];
      ih[o] = s;
      let t = b_hh.data[o]; const rowH = o * w_hh.shape[1];
      for (let i = 0; i < H; i++) t += h[i] * w_hh.data[rowH + i];
      hh[o] = t;
    }
    // rz = sigmoid(rz_ih + rz_hh); r,z = split(rz); n = tanh(n_ih + r*n_hh)
    const out = new Float64Array(H);
    for (let i = 0; i < H; i++) {
      const r = sigmoid(ih[i] + hh[i]);
      const z = sigmoid(ih[H + i] + hh[H + i]);
      const n = Math.tanh(ih[2 * H + i] + r * hh[2 * H + i]);
      out[i] = (1 - z) * n + z * h[i];
    }
    return out;
  }

  private embed(emb: Weight, idx: number): Float32Array { // row idx of (V, E)
    const E = emb.shape[1];
    return emb.data.subarray(idx * E, idx * E + E);
  }

  predict(word: string): string[] {
    const chars = [...word, '</s>'];
    const ids = chars.map((c) => (this.g2idx[c] ?? this.g2idx['<unk>']));
    // encoder
    let h: Float64Array = new Float64Array(this.H);
    for (const id of ids) h = this.grucell(this.embed(this.w.enc_emb, id), h,
      this.w.enc_w_ih, this.w.enc_w_hh, this.w.enc_b_ih, this.w.enc_b_hh);
    // decoder, start token <s>=2
    let dec: ArrayLike<number> = this.embed(this.w.dec_emb, 2);
    const preds: number[] = [];
    for (let step = 0; step < 20; step++) {
      h = this.grucell(dec, h, this.w.dec_w_ih, this.w.dec_w_hh, this.w.dec_b_ih, this.w.dec_b_hh);
      // logits = h @ fc_w.T + fc_b ; argmax
      const V = this.w.fc_w.shape[0];
      let best = 0, bestv = -Infinity;
      for (let o = 0; o < V; o++) {
        let s = this.w.fc_b.data[o]; const row = o * this.H;
        for (let i = 0; i < this.H; i++) s += h[i] * this.w.fc_w.data[row + i];
        if (s > bestv) { bestv = s; best = o; }
      }
      if (best === 3) break; // </s>
      preds.push(best);
      dec = this.embed(this.w.dec_emb, best);
    }
    return preds.map((i) => this.idx2p[i] ?? '<unk>');
  }

  // tokenize: input is already restricted to [a-z '.,?!\- ] after preprocessing
  private tokenize(text: string): string[] {
    return text.match(/[a-z][a-z'\-]*|[.,?!]/g) || [];
  }

  private preprocess(text: string): string {
    text = normalizeNumbers(text);
    // strip accents (NFD, drop combining marks)
    text = text.normalize('NFD').replace(/[̀-ͯ]/g, '');
    text = text.toLowerCase();
    text = text.replace(/[^ a-z'.,?!\-]/g, '');
    return text.replace(/i\.e\./g, 'that is').replace(/e\.g\./g, 'for example');
  }

  // word tokens (charsiu's _get_words): same preprocessing + tokenization
  textToWords(text: string): string[] {
    return this.tokenize(this.preprocess(text));
  }

  convert(text: string): string[] {
    text = this.preprocess(text);
    const words = this.tokenize(text);
    const prons: string[] = [];
    for (const word of words) {
      let pron: string[];
      if (!/[a-z]/.test(word)) pron = [word];
      else if (this.homographs[word]) pron = this.homographs[word][1]; // default pron (no POS yet)
      else if (this.cmu[word]) pron = this.cmu[word].split(' ');
      else pron = this.predict(word);
      prons.push(...pron, ' ');
    }
    prons.pop();
    return prons;
  }
}

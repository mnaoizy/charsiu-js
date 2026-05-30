// Mandarin grapheme-to-phoneme, ported from g2pM (kakaobrain).
// A 1-layer BiLSTM + 2-layer FC disambiguates polyphonic characters; everything
// else is a CEDICT dictionary lookup. Output: pinyin syllables with tone digits,
// matching G2pM(sent) (char_split=False) exactly.
import type { G2pmAssets, Weight } from './types.js';

const UNK = '<UNK>', BOS = '시', EOS = '끝', SPLIT = '▁';
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export class G2pM {
  private cedict: Record<string, string[]>;
  private char2idx: Record<string, number>;
  private idx2class: string[];
  private w: Record<string, Weight> = {};
  private H: number;
  private E: number;

  constructor(assets: G2pmAssets) {
    this.cedict = assets.cedict;
    this.char2idx = assets.char2idx;
    this.idx2class = assets.idx2class;
    const { manifest, data } = assets.lstm;
    for (const [k, { offset, shape }] of Object.entries(manifest)) {
      const len = shape.reduce((a, b) => a * b, 1);
      this.w[k] = { data: data.subarray(offset, offset + len), shape };
    }
    this.H = this.w['lstm.weight_hh_l0'].shape[1]; // 32
    this.E = this.w['embedding.weight'].shape[1];  // 64
  }

  private emb(idx: number): Float32Array { const E = this.E; return this.w['embedding.weight'].data.subarray(idx * E, idx * E + E); }

  // y = W x + b ; W flat (out,in)
  private affine(W: Weight, b: Weight, x: ArrayLike<number>): Float64Array {
    const out = W.shape[0], inn = W.shape[1], y = new Float64Array(out);
    for (let o = 0; o < out; o++) { let s = b.data[o]; const r = o * inn; for (let i = 0; i < inn; i++) s += W.data[r + i] * x[i]; y[o] = s; }
    return y;
  }

  // one LSTM step; gate layout [i, f, g, o] (PyTorch). dir: '' or '_reverse'
  private lstmCell(x: ArrayLike<number>, prevH: ArrayLike<number>, prevC: ArrayLike<number>, dir: '' | '_reverse'): [Float64Array, Float64Array] {
    const H = this.H;
    const ih = this.affine(this.w[`lstm.weight_ih_l0${dir}`], this.w[`lstm.bias_ih_l0${dir}`], x);
    const hh = this.affine(this.w[`lstm.weight_hh_l0${dir}`], this.w[`lstm.bias_hh_l0${dir}`], prevH);
    const h = new Float64Array(H), c = new Float64Array(H);
    for (let k = 0; k < H; k++) {
      const i = sigmoid(ih[k] + hh[k]);
      const f = sigmoid(ih[H + k] + hh[H + k]);
      const g = Math.tanh(ih[2 * H + k] + hh[2 * H + k]);
      const o = sigmoid(ih[3 * H + k] + hh[3 * H + k]);
      c[k] = f * prevC[k] + i * g;
      h[k] = o * Math.tanh(c[k]);
    }
    return [h, c];
  }

  private fc(x: ArrayLike<number>): number { // [2H] -> relu(W0 x + b0) -> W1 . + b1 -> argmax
    const h0 = this.affine(this.w['logit_layer.0.weight'], this.w['logit_layer.0.bias'], x);
    for (let i = 0; i < h0.length; i++) if (h0[i] < 0) h0[i] = 0;
    const logits = this.affine(this.w['logit_layer.2.weight'], this.w['logit_layer.2.bias'], h0);
    let best = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }

  // run BiLSTM over inputIds, classify at each targetIdx -> class ids
  private predict(inputIds: number[], targetIdx: number[]): number[] {
    const L = inputIds.length, H = this.H;
    const fwH: Float64Array[] = [];
    let h: Float64Array = new Float64Array(H), c: Float64Array = new Float64Array(H);
    for (let t = 0; t < L; t++) { [h, c] = this.lstmCell(this.emb(inputIds[t]), h, c, ''); fwH.push(h); }
    h = new Float64Array(H); c = new Float64Array(H);
    const bwRev: Float64Array[] = [];
    for (let t = 0; t < L; t++) { [h, c] = this.lstmCell(this.emb(inputIds[L - 1 - t]), h, c, '_reverse'); bwRev.push(h); }
    // reverse back so index t aligns with position t
    const bwBack = bwRev.slice().reverse();
    return targetIdx.map((ti) => {
      const cat = new Float64Array(2 * H);
      cat.set(fwH[ti], 0); cat.set(bwBack[ti], H);
      return this.fc(cat);
    });
  }

  // sent: string of Chinese characters. Returns pinyin syllables (tone digits).
  convert(sent: string, { tone = true }: { tone?: boolean } = {}): string[] {
    const chars = [...sent];
    const inputIds: number[] = [], polyIdx: number[] = [], pros: string[] = [];
    for (let idx = 0; idx < chars.length; idx++) {
      const ch = chars[idx];
      inputIds.push(this.char2idx[ch] ?? this.char2idx[UNK]);
      const prons = this.cedict[ch];
      if (prons) {
        if (prons.length > 1) { polyIdx.push(idx); pros.push(SPLIT); }
        else pros.push(tone ? prons[0] : prons[0].slice(0, -1));
      } else pros.push(ch);
    }
    if (polyIdx.length) {
      const ids = [this.char2idx[BOS], ...inputIds, this.char2idx[EOS]];
      const preds = this.predict(ids, polyIdx.map((i) => i + 1));
      for (let k = 0; k < polyIdx.length; k++) {
        let pron = this.idx2class[preds[k]];
        if (!tone) pron = pron.slice(0, -1);
        pros[polyIdx[k]] = pron;
      }
    }
    return this.join(pros);
  }

  // char_split=False join: pinyin syllables delimited by '|', single chars kept inline
  private join(pros: string[]): string[] {
    let s = '';
    const d = '|';
    for (let pro of pros) {
      if (pro.length === 1) s += pro;
      else { if (s.length > 0 && s[s.length - 1] !== d) pro = d + pro; s += pro + d; }
    }
    if (s[s.length - 1] === d) s = s.slice(0, -1);
    return s.split(d);
  }
}

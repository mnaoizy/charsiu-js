// Japanese phonemizer: morphological analysis (tokana) + kana->phoneme (./g2p-ja)
// -> the phone-id sequence the CTC aligner forces against, plus word grouping for
// word-level alignment. Targets the prj-beatrice/japanese-hubert-base-phoneme-ctc
// phone set (see ./g2p-ja.ts and assets/vocab_ja.json).
import { kanaToPhonemes } from './g2p-ja.js';
import type { CtcPhonemizerLike } from './types.js';

/** Minimal shape of a tokana IPADIC token we read. */
export interface JaToken {
  surface: string;
  /** 発音 (pronunciation reading, katakana). */
  pronunciation: string;
  /** 読み (dictionary reading, katakana). */
  reading: string;
  pos: string;
}

/** Minimal shape of a tokana tokenizer. */
export interface TokanaLike {
  tokenize(text: string): JaToken[];
}

const KATAKANA = /^[ァ-ヶー・ー]+$/;            // pure-katakana surface = loanword
// Phones the model's 47-class vocab doesn't have -> nearest in-vocab fallback.
const OOV: Record<string, string> = { ty: 'ch', sy: 'sh', cy: 'ch', dy: 'dy' };

// Sino-Japanese (kanji-read) words pronounce え-row + イ as a long え (e.g. 学生
// セイ -> s e e). IPADIC's reading keeps セイ, so lengthen e+i -> e+e here. Only
// applied to non-katakana tokens, so loanwords (エイト e i t o) are untouched.
function lengthenEi(phones: string[]): string[] {
  const out = phones.slice();
  for (let i = 0; i + 1 < out.length; i++) if (out[i] === 'e' && out[i + 1] === 'i') out[i + 1] = 'e';
  return out;
}

export class PhonemizerJa implements CtcPhonemizerLike {
  readonly vocab: Record<string, number>;
  readonly id2phone: Record<string, string>;
  readonly blankIdx: number;
  readonly silIdx: number;
  private warned = new Set<string>();

  constructor(private tokenizer: TokanaLike, vocab: Record<string, number>) {
    this.vocab = vocab;
    this.id2phone = {};
    for (const [p, i] of Object.entries(vocab)) this.id2phone[String(i)] = p;
    this.blankIdx = vocab.PAD ?? 0;
    this.silIdx = vocab.sil ?? vocab.pau ?? -1;
  }

  /** text -> { groups: phones per word, words: surface forms }. */
  getPhonesAndWords(text: string): { groups: string[][]; words: string[] } {
    const groups: string[][] = [];
    const words: string[] = [];
    for (const t of this.tokenizer.tokenize(text)) {
      const kana = t.pronunciation && t.pronunciation !== '*' ? t.pronunciation
        : (t.reading && t.reading !== '*' ? t.reading : '');
      if (!kana) continue;                       // punctuation / symbols
      let phones = kanaToPhonemes(kana);
      if (!KATAKANA.test(t.surface)) phones = lengthenEi(phones);
      if (!phones.length) continue;
      groups.push(phones);
      words.push(t.surface);
    }
    return { groups, words };
  }

  /** Flatten phone groups to model ids, mapping out-of-vocab phones to the
   *  nearest in-vocab phone (or dropping if truly unknown). */
  getTargetIds(groups: string[][]): number[] {
    const ids: number[] = [];
    for (const g of groups) for (const p of g) {
      let id: number | undefined = this.vocab[p];
      if (id === undefined && OOV[p] !== undefined) id = this.vocab[OOV[p]];
      if (id === undefined) {
        if (!this.warned.has(p)) { console.warn(`[charsiu-js] dropping out-of-vocab phone: ${p}`); this.warned.add(p); }
        continue;
      }
      ids.push(id);
    }
    return ids;
  }

  getPhoneIds(text: string): number[] {
    return this.getTargetIds(this.getPhonesAndWords(text).groups);
  }

  /** text -> the CTC target plus per-word phone grouping (for word alignment). */
  phonemize(text: string): { groups: string[][]; words: string[]; targetIds: number[]; groupLens: number[] } {
    const { groups, words } = this.getPhonesAndWords(text);
    const groupIds = groups.map((g) => this.getTargetIds([g]));
    return { groups, words, targetIds: groupIds.flat(), groupLens: groupIds.map((a) => a.length) };
  }
}

// English text -> phone ids, ported from charsiu's CharsiuPreprocessor_en
// (get_phones_and_words + get_phone_ids). Mirrors the Python exactly:
//   g2p -> group phones per word -> drop punctuation -> strip stress ->
//   map to ids via the tokenizer_en_cmu vocab -> pad with [SIL] at both ends.
import type { G2p } from './g2p.js';
import type { Segment, PhonemizerLike } from './types.js';

const SIL = '[SIL]';
const stripStress = (p: string) => p.replace(/\d/g, '');

export class PhonemizerEn implements PhonemizerLike {
  silIdx: number;
  id2phone: Record<string, string> = {};
  private g2p: G2p;
  private vocab: Record<string, number>;

  // vocab: { phone: id } from tokenizer_en_cmu (assets/vocab_en.json)
  constructor(g2p: G2p, vocab: Record<string, number>) {
    this.g2p = g2p;
    this.vocab = vocab;
    this.silIdx = vocab[SIL];
    for (const [p, i] of Object.entries(vocab)) this.id2phone[String(i)] = p;
  }

  // -> { groups: per-word phone groups (stress kept), words } with punctuation dropped
  getPhonesAndWords(text: string): { groups: string[][]; words: string[] } {
    const flat = this.g2p.convert(text);      // phone tokens incl. ' ' separators
    const words = this.g2p.textToWords(text); // one token per word/punctuation
    const allGroups: string[][] = [];
    let cur: string[] = [];
    for (const tok of flat) {
      if (tok === ' ') { if (cur.length) allGroups.push(cur); cur = []; }
      else cur.push(tok);
    }
    if (cur.length) allGroups.push(cur);
    // zip groups with words; keep only word groups (\w); punctuation set is empty -> dropped
    const groups: string[][] = [], keptWords: string[] = [];
    const n = Math.min(allGroups.length, words.length);
    for (let i = 0; i < n; i++) {
      if (/\w/.test(allGroups[i][0])) { groups.push(allGroups[i]); keptWords.push(words[i]); }
    }
    return { groups, words: keptWords };
  }

  // map phone-level segments to word-level segments (port of align_words)
  alignWords(preds: Segment[], groups: string[][], words: string[]): Segment[] {
    const wordsRep: string[] = [], phonesRep: string[] = [];
    for (let i = 0; i < groups.length; i++)
      for (const p of groups[i]) { wordsRep.push(words[i]); phonesRep.push(stripStress(p)); }

    const wordDur: [Segment, string][] = [];
    let count = 0;
    for (const dur of preds) {
      if (dur[2] === SIL) { wordDur.push([dur, SIL]); continue; }
      while (count < phonesRep.length && dur[2] !== phonesRep[count]) count++;
      wordDur.push([dur, wordsRep[count]]);
    }
    // merge consecutive same-word phone segments into word segments
    const out: Segment[] = [];
    for (let i = 0; i < wordDur.length; ) {
      let j = i; while (j < wordDur.length && wordDur[j][1] === wordDur[i][1]) j++;
      out.push([wordDur[i][0][0], wordDur[j - 1][0][1], wordDur[i][1]]);
      i = j;
    }
    return out;
  }

  getPhoneIds(groups: string[][]): number[] {
    const flat = groups.flat();
    const ids = flat.map((p) => this.vocab[stripStress(p)]);
    if (ids[0] !== this.silIdx) ids.unshift(this.silIdx);
    if (ids[ids.length - 1] !== this.silIdx) ids.push(this.silIdx);
    return ids;
  }
}

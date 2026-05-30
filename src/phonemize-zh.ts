// Mandarin text -> phone ids, ported from charsiu's CharsiuPreprocessor_zh.
// g2pM -> pinyin -> transform table -> split into initial + final phones
// (with er/rhyme mappings) -> map to tokenizer_zh_pinyin ids -> pad with [SIL].
import type { G2pM } from './g2pm.js';
import type { Segment, PhonemizerLike } from './types.js';

const SIL = '[SIL]';

const CONSONANTS = new Set(['b','p','m','f','d','t','n','l','g','k','h','j','q','x',
  'zh','ch','sh','r','z','c','s']);

const TRANSFORM: Record<string, string> = { ju:'jv', qu:'qv', xu:'xv', jue:'jve', que:'qve', xue:'xve',
  quan:'qvan', xuan:'xvan', juan:'jvan', qun:'qvn', xun:'xvn', jun:'jvn',
  yuan:'van', yue:'ve', yun:'vn', you:'iou', yan:'ian', yin:'in',
  wa:'ua', wo:'uo', wai:'uai', weng:'ueng', wang:'uang', wu:'u',
  yu:'v', yi:'i', yo:'io', ya:'ia', ye:'ie', yao:'iao', yang:'iang',
  ying:'ing', yong:'iong', yvan:'van', yve:'ve', yvn:'vn',
  wei:'ui', wan:'uan', wen:'un', yv:'v', wuen:'un', wuo:'uo', wuang:'uang',
  wuan:'uan', wua:'ua', wuai:'uai',
  zhi:'zhiii', chi:'chiii', shi:'shiii', zi:'zii', ci:'cii', si:'sii' };

const ER: Record<string, string[]> = { er1:['e1','rr'], er2:['e2','rr'], er3:['e3','rr'],
  er4:['e4','rr'], er5:['e5','rr'], r5:['e5','rr'] };

const RHYME: Record<string, string | string[]> = { iu1:'iou1', iu2:'iou2', iu3:'iou3', iu4:'iou4', iu5:'iou5',
  'u:e1':'ve1','u:e2':'ve2','u:e3':'ve3','u:e4':'ve4','u:e5':'ve5',
  'u:1':'v1','u:2':'v2','u:3':'v3','u:4':'v4','u:5':'v5',
  ueng1:['u1','eng1'], ueng2:['u2','eng2'], ueng3:['u3','eng3'],
  ueng4:['u4','eng4'], ueng5:['u5','eng5'], io5:['i5','o5'], io4:['i4','o4'], io1:['i1','o1'] };

const rhyme = (s: string): string | string[] => RHYME[s] ?? s;

function separateSyllable(syl: string): string[] {
  if (syl === 'ri4') return ['r', 'iii4'];
  const body = syl.slice(0, -1);
  if (body === 'ueng' || body === 'io') { const r = rhyme(syl); return Array.isArray(r) ? r : [r]; }
  if (ER[syl]) return ER[syl];
  if (CONSONANTS.has(syl.slice(0, 2))) { const r = rhyme(syl.slice(2)); return [syl.slice(0, 2), ...(Array.isArray(r) ? r : [r])]; }
  if (CONSONANTS.has(syl[0])) { const r = rhyme(syl.slice(1)); return [syl[0], ...(Array.isArray(r) ? r : [r])]; }
  return [syl];
}

export class PhonemizerZh implements PhonemizerLike {
  silIdx: number;
  id2phone: Record<string, string> = {};
  private g2p: G2pM;
  private vocab: Record<string, number>;

  // g2pM instance + vocab { phone: id } from tokenizer_zh_pinyin
  constructor(g2p: G2pM, vocab: Record<string, number>) {
    this.g2p = g2p;
    this.vocab = vocab;
    this.silIdx = vocab[SIL];
    for (const [p, i] of Object.entries(vocab)) this.id2phone[String(i)] = p;
  }

  // -> { groups: per-syllable phone arrays, words: per-syllable hanzi }
  getPhonesAndWords(text: string): { groups: string[][]; words: string[] } {
    const phones = this.g2p.convert(text); // one pinyin per character
    const chars = [...text];
    const groups: string[][] = [], words: string[] = [];
    const n = Math.min(phones.length, chars.length);
    for (let i = 0; i < n; i++) {
      const p = phones[i];
      if (/\w+:?\d/.test(p)) {
        const body = p.slice(0, -1), tone = p.slice(-1);
        groups.push(separateSyllable((TRANSFORM[body] ?? body) + tone));
        words.push(chars[i]);
      } // punctuation set is empty -> non-tonal entries dropped
    }
    return { groups, words };
  }

  getPhoneIds(groups: string[][]): number[] {
    const ids = groups.flat().map((p) => this.vocab[p]);
    if (ids[0] !== this.silIdx) ids.unshift(this.silIdx);
    if (ids[ids.length - 1] !== this.silIdx) ids.push(this.silIdx);
    return ids;
  }

  // phone segments -> word (hanzi) segments; same scheme as English align_words
  alignWords(preds: Segment[], groups: string[][], words: string[]): Segment[] {
    const wordsRep: string[] = [], phonesRep: string[] = [];
    for (let i = 0; i < groups.length; i++)
      for (const p of groups[i]) { wordsRep.push(words[i]); phonesRep.push(p); }
    const wordDur: [Segment, string][] = [];
    let count = 0;
    for (const dur of preds) {
      if (dur[2] === SIL) { wordDur.push([dur, SIL]); continue; }
      while (count < phonesRep.length && dur[2] !== phonesRep[count]) count++;
      wordDur.push([dur, wordsRep[count]]);
    }
    const out: Segment[] = [];
    for (let i = 0; i < wordDur.length; ) {
      let j = i; while (j < wordDur.length && wordDur[j][1] === wordDur[i][1]) j++;
      out.push([wordDur[i][0][0], wordDur[j - 1][0][1], wordDur[i][1]]);
      i = j;
    }
    return out;
  }
}

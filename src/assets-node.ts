// Node asset loader for the English + Mandarin g2p and aligner.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { G2pAssets, G2pmAssets } from './types.js';

const A = (name: string) => fileURLToPath(new URL(`../assets/${name}`, import.meta.url));
const readJson = (name: string) => JSON.parse(readFileSync(A(name), 'utf8'));
function readFloat32(name: string): Float32Array {
  const buf = readFileSync(A(name));
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

export function loadG2pAssets(): G2pAssets {
  return {
    cmudict: readJson('cmudict.json'),
    homographs: readJson('homographs.json'),
    vocab: readJson('g2p_vocab.json'),
    gru: { manifest: readJson('g2p_gru.json'), data: readFloat32('g2p_gru.bin') },
  };
}

// tokenizer_en_cmu phone->id vocab (for phonemize + alignment)
export function loadPhoneVocab(): Record<string, number> {
  return readJson('vocab_en.json');
}

// tokenizer_zh_pinyin phone->id vocab
export function loadPhoneVocabZh(): Record<string, number> {
  return readJson('vocab_zh.json');
}

// g2pM (Mandarin) assets
export function loadG2pmAssets(): G2pmAssets {
  return {
    cedict: readJson('g2pm_cedict.json'),
    char2idx: readJson('g2pm_char2idx.json'),
    idx2class: readJson('g2pm_idx2class.json'),
    lstm: { manifest: readJson('g2pm_lstm.json'), data: readFloat32('g2pm_lstm.bin') },
  };
}

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

// Japanese (OpenJTalk) phone->id vocab, matching the hubert phoneme-CTC model.
export function loadPhoneVocabJa(): Record<string, number> {
  return readJson('vocab_ja.json');
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

// Read a 16 kHz mono 16-bit PCM WAV file into Float32 samples in [-1, 1] — the
// Node counterpart to the browser's decodeToMono16k. It does NOT resample: the
// model wants 16 kHz, so a different rate throws (resample with your own tool
// first). Multi-channel input is downmixed to mono.
export function loadWav16k(path: string | URL): Float32Array {
  const buf = readFileSync(path);
  if (buf.length < 44 || buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE')
    throw new Error('not a WAV file');
  let off = 12, channels = 1, sampleRate = 0, bits = 16, dataOff = -1, dataLen = 0;
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === 'fmt ') {
      channels = buf.readUInt16LE(off + 10);
      sampleRate = buf.readUInt32LE(off + 12);
      bits = buf.readUInt16LE(off + 22);
    } else if (id === 'data') { dataOff = off + 8; dataLen = size; }
    off += 8 + size + (size & 1);
  }
  if (dataOff < 0) throw new Error('WAV has no data chunk');
  if (bits !== 16) throw new Error(`expected 16-bit PCM WAV, got ${bits}-bit`);
  if (sampleRate !== 16000)
    throw new Error(`expected 16 kHz audio, got ${sampleRate} Hz (resample to 16 kHz first; charsiu-js does not resample in Node)`);
  const frames = Math.floor(dataLen / 2 / channels);
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let s = 0;
    for (let c = 0; c < channels; c++) s += buf.readInt16LE(dataOff + (i * channels + c) * 2);
    out[i] = s / channels / 32768;
  }
  return out;
}

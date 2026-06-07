// Node convenience factory: wires onnxruntime-node + bundled g2p assets together.
// Re-exports the runtime-agnostic core so Node users can `import ... from 'charsiu-js'`.
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';
import { mkdir, writeFile, access } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { G2p } from './g2p.js';
import { G2pM } from './g2pm.js';
import { PhonemizerEn } from './phonemize-en.js';
import { PhonemizerZh } from './phonemize-zh.js';
import { PhonemizerJa } from './phonemize-ja.js';
import type { TokanaLike } from './phonemize-ja.js';
import { ForcedAligner } from './index.js';
import { CtcForcedAligner } from './aligner-ctc.js';
import type { AlignSession, OrtLike } from './types.js';
import { loadG2pAssets, loadPhoneVocab, loadG2pmAssets, loadPhoneVocabZh, loadPhoneVocabJa } from './assets-node.js';

export * from './index.js';
export { G2pM } from './g2pm.js';
export { PhonemizerZh } from './phonemize-zh.js';
export { PhonemizerJa } from './phonemize-ja.js';

// Local model produced by `npm run convert` (present in a dev checkout only).
const LOCAL_MODEL = fileURLToPath(new URL('../models/en_w2v2_fc_10ms/model_quantized.onnx', import.meta.url));
const LOCAL_MODEL_ZH = fileURLToPath(new URL('../models/zh_w2v2_tiny_fc_10ms/model_quantized.onnx', import.meta.url));
const LOCAL_MODEL_JA = fileURLToPath(new URL('../models/japanese-hubert-base-phoneme-ctc/model_quantized.onnx', import.meta.url));
// Default hosted models (downloaded + cached on first use).
const HF = 'https://huggingface.co/mnaoizyyy/charsiu-js-models/resolve/main';
const DEFAULT_MODEL_URL = `${HF}/en_w2v2_fc_10ms/model_quantized.onnx`;
const DEFAULT_MODEL_URL_ZH = `${HF}/zh_w2v2_tiny_fc_10ms/model_quantized.onnx`;
const DEFAULT_MODEL_URL_JA = `${HF}/japanese-hubert-base-phoneme-ctc/model_quantized.onnx`;
const DEFAULT_CACHE = join(homedir(), '.cache', 'charsiu-js');

export interface NodeAlignerOptions {
  /** Path to a local .onnx model. */
  modelPath?: string;
  /** URL to download the .onnx model from (cached locally). */
  modelUrl?: string;
  /** Cache directory for downloaded models. Default ~/.cache/charsiu-js. */
  cacheDir?: string;
}

async function downloadToCache(url: string, cacheDir: string): Promise<string> {
  // include the parent path segment so e.g. en/model_quantized.onnx and
  // zh/model_quantized.onnx don't collide in the cache.
  let name: string;
  try {
    const parts = new URL(url).pathname.split('/').filter(Boolean);
    name = parts.slice(-2).join('__') || 'model.onnx';
  } catch { name = url.split('/').pop()?.split('?')[0] || 'model.onnx'; }
  const file = join(cacheDir, name);
  try { await access(file); return file; } catch { /* not cached yet */ }
  await mkdir(cacheDir, { recursive: true });
  const res = await fetch(url); // follows redirects (HF Hub -> CDN)
  if (!res.ok) throw new Error(`failed to download model: ${url} (${res.status})`);
  await writeFile(file, Buffer.from(await res.arrayBuffer()));
  return file;
}

// Resolution order: explicit modelPath -> explicit modelUrl -> bundled local
// model (dev checkout) -> the default hosted model (downloaded + cached).
async function resolveModel(opts: NodeAlignerOptions, localDefault: string, defaultUrl: string): Promise<string> {
  const { modelPath, modelUrl, cacheDir = DEFAULT_CACHE } = opts;
  if (modelPath) return modelPath;
  if (modelUrl) return downloadToCache(modelUrl, cacheDir);
  if (localDefault && existsSync(localDefault)) return localDefault;
  return downloadToCache(defaultUrl, cacheDir);
}

/** English aligner wired to onnxruntime-node + bundled assets. */
export async function createNodeAligner(opts: NodeAlignerOptions = {}): Promise<ForcedAligner> {
  const ort = await import('onnxruntime-node');
  const phonemizer = new PhonemizerEn(new G2p(loadG2pAssets()), loadPhoneVocab());
  const session = await ort.InferenceSession.create(await resolveModel(opts, LOCAL_MODEL, DEFAULT_MODEL_URL));
  return new ForcedAligner({ session: session as unknown as AlignSession, ort: ort as unknown as OrtLike, phonemizer });
}

/** Mandarin aligner wired to onnxruntime-node + bundled assets. */
export async function createNodeAlignerZh(opts: NodeAlignerOptions = {}): Promise<ForcedAligner> {
  const ort = await import('onnxruntime-node');
  const phonemizer = new PhonemizerZh(new G2pM(loadG2pmAssets()), loadPhoneVocabZh());
  const session = await ort.InferenceSession.create(await resolveModel(opts, LOCAL_MODEL_ZH, DEFAULT_MODEL_URL_ZH));
  return new ForcedAligner({ session: session as unknown as AlignSession, ort: ort as unknown as OrtLike, phonemizer });
}

export interface NodeAlignerJaOptions extends NodeAlignerOptions {
  /** Path to a tokana-compiled IPADIC dictionary directory (see
   *  `npx tokana build <mecab-ipadic-src> <dicPath>`). Required. */
  dicPath: string;
  /** tokana dictionary format. Default 'ipadic'. */
  dictFormat?: 'ipadic' | 'unidic' | 'neologd';
}

/**
 * Japanese aligner wired to onnxruntime-node + tokana + bundled assets.
 * The acoustic model (prj-beatrice/japanese-hubert-base-phoneme-ctc, CTC, 20 ms,
 * raw waveform) auto-downloads from the Hub on first use, like EN/ZH; or pass
 * `modelPath`/`modelUrl`. Requires `tokana` (peer dep) and a compiled dictionary
 * at `dicPath` (build one with `npm run setup-dict`).
 */
export async function createNodeAlignerJa(opts: NodeAlignerJaOptions): Promise<CtcForcedAligner> {
  const ort = await import('onnxruntime-node');
  const { createTokenizer } = await import('tokana');
  const tokenizer = await createTokenizer({ format: opts.dictFormat ?? 'ipadic', dicPath: opts.dicPath });
  // IPADIC/NEologd tokens carry surface/pronunciation/reading/pos (TokanaLike);
  // UniDic uses different fields and isn't supported by PhonemizerJa.
  const phonemizer = new PhonemizerJa(tokenizer as unknown as TokanaLike, loadPhoneVocabJa());
  const session = await ort.InferenceSession.create(await resolveModel(opts, LOCAL_MODEL_JA, DEFAULT_MODEL_URL_JA));
  return new CtcForcedAligner({ session: session as unknown as AlignSession, ort: ort as unknown as OrtLike, phonemizer });
}

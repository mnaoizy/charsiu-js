import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import { G2pM } from '../dist/g2pm.js';
import { PhonemizerZh } from '../dist/phonemize-zh.js';
import { ForcedAligner } from '../dist/index.js';
import { loadG2pmAssets, loadWav16k } from '../dist/assets-node.js';
import { fileURLToPath } from 'node:url';

const root = new URL('../', import.meta.url);
const A = (f) => fileURLToPath(new URL(`assets/${f}`, root));
const text = readFileSync(new URL('sample/zh_transcript.txt', root), 'utf8').trim();
const wav = loadWav16k(new URL('sample/zh_sample.wav', root));
const oracle = JSON.parse(readFileSync(new URL('sample/zh_align_oracle.json', root)));
const oracleW = JSON.parse(readFileSync(new URL('sample/zh_align_oracle_words.json', root)));
const spec = JSON.parse(readFileSync(new URL('sample/zh_align_spec.json', root)));

const vocab = JSON.parse(readFileSync(A('vocab_zh.json')));
const phonemizer = new PhonemizerZh(new G2pM(loadG2pmAssets()), vocab);
const session = await ort.InferenceSession.create(new URL('models/zh_w2v2_tiny_fc_10ms/model.onnx', root).pathname);
const aligner = new ForcedAligner({ session, ort, phonemizer });

console.log('text:', text);
const t0 = performance.now();
const { phones, words, phoneIds } = await aligner.align(wav, text);
console.log(`aligned in ${(performance.now()-t0).toFixed(0)}ms`);

const idsOk = phoneIds.length===spec.phone_ids.length && phoneIds.every((v,i)=>v===spec.phone_ids[i]);
const seqOk = phones.length===oracle.length && phones.every((s,i)=>s[2]===oracle[i][2]);
let md=0; for(let i=0;i<oracle.length;i++) md=Math.max(md,Math.abs(phones[i][0]-oracle[i][0]),Math.abs(phones[i][1]-oracle[i][1]));
const wOk = words.length===oracleW.length && words.every((s,i)=>s[2]===oracleW[i][2] && Math.abs(s[0]-oracleW[i][0])<0.02 && Math.abs(s[1]-oracleW[i][1])<0.02);

console.log('phone ids match:', idsOk?'OK':'MISMATCH');
console.log('phones:', phones.map(p=>p[2]).filter(p=>p!=='[SIL]').join(' '));
console.log('phones match oracle:', seqOk?'OK':'MISMATCH', '| max boundary diff:', md.toFixed(2)+'s');
console.log('words:', words.map(w=>w[2]).join(' '));
console.log('words match oracle:', wOk?'OK':'MISMATCH');
const pass = idsOk && seqOk && md<=0.02 && wOk;
console.log(`\n${pass?'PASS: standalone JS Mandarin alignment matches charsiu':'FAIL'}`);
process.exit(pass?0:1);

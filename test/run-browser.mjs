// Headless browser E2E: serve the project, drive the demo in Chrome, and verify
// the client-side alignment matches the charsiu oracle.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { extname } from 'node:path';
import { chromium } from 'playwright';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const MIME = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.wav': 'audio/wav',
  '.onnx': 'application/octet-stream', '.bin': 'application/octet-stream', '.css': 'text/css' };

const server = createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(req.url.split('?')[0]);
    const data = await readFile(ROOT + path.replace(/^\//, ''));
    res.setHeader('Content-Type', MIME[extname(path)] || 'application/octet-stream');
    res.end(data);
  } catch { res.statusCode = 404; res.end('not found'); }
});

await new Promise((r) => server.listen(0, r));
const port = server.address().port;
const base = `http://localhost:${port}`;
console.log(`serving ${ROOT} at ${base}`);

const oracle = JSON.parse(await readFile(new URL('../sample/align_oracle.json', import.meta.url)));
const text = (await readFile(new URL('../sample/transcript.txt', import.meta.url), 'utf8')).trim();
const oracleZh = JSON.parse(await readFile(new URL('../sample/zh_align_oracle.json', import.meta.url)));
const textZh = (await readFile(new URL('../sample/zh_transcript.txt', import.meta.url), 'utf8')).trim();

// compare browser phones to an oracle: returns { seqMatch, maxDiff }
function compare(phones, oracleSegs) {
  const seqMatch = phones.length === oracleSegs.length && phones.every((s, i) => s[2] === oracleSegs[i][2]);
  let maxDiff = 0;
  for (let i = 0; i < oracleSegs.length; i++)
    maxDiff = Math.max(maxDiff, Math.abs(phones[i][0] - oracleSegs[i][0]), Math.abs(phones[i][1] - oracleSegs[i][1]));
  return { seqMatch, maxDiff };
}

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage();
page.on('console', (m) => console.log('  [page]', m.text()));
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));

let pass = false;
try {
  await page.goto(`${base}/web/index.html`);
  console.log('loading model + assets in browser…');
  await page.waitForFunction('window.__ready', null, { timeout: 120000 });

  // English
  console.log('aligner ready, running English alignment…');
  const en = await page.evaluate(async (t) => window.__alignUrl(t, '/sample/sample.wav', 'en'), text);
  const enCmp = compare(en.phones, oracle);
  console.log(`  EN phones: ${en.phones.map((p) => p[2]).filter((p) => p !== '[SIL]').join(' ')}`);
  console.log(`  EN words:  ${en.words.map((w) => w[2]).join(' ')}`);
  console.log(`  EN match oracle: ${enCmp.seqMatch ? 'OK' : 'MISMATCH'} | max boundary diff: ${enCmp.maxDiff.toFixed(2)}s`);

  // Mandarin (loads the zh model + g2pM assets on demand)
  console.log('running Mandarin alignment…');
  const zh = await page.evaluate(async (t) => window.__alignUrl(t, '/sample/zh_sample.wav', 'zh'), textZh);
  const zhCmp = compare(zh.phones, oracleZh);
  console.log(`  ZH phones: ${zh.phones.map((p) => p[2]).filter((p) => p !== '[SIL]').join(' ')}`);
  console.log(`  ZH words:  ${zh.words.map((w) => w[2]).join(' ')}`);
  console.log(`  ZH match oracle: ${zhCmp.seqMatch ? 'OK' : 'MISMATCH'} | max boundary diff: ${zhCmp.maxDiff.toFixed(2)}s`);

  pass = enCmp.seqMatch && enCmp.maxDiff <= 0.03 && zhCmp.seqMatch && zhCmp.maxDiff <= 0.03;
} finally {
  await browser.close();
  server.close();
}

console.log(`\n${pass ? 'PASS: in-browser (onnxruntime-web) EN + ZH alignment matches charsiu' : 'FAIL'}`);
process.exit(pass ? 0 : 1);

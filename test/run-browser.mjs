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

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage();
page.on('console', (m) => console.log('  [page]', m.text()));
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));

let pass = false;
try {
  await page.goto(`${base}/web/index.html`);
  console.log('loading model + assets in browser…');
  await page.waitForFunction('window.__ready', null, { timeout: 120000 });
  console.log('aligner ready, running alignment…');
  const result = await page.evaluate(async (t) => window.__alignUrl(t, '/sample/sample.wav'), text);

  const phones = result.phones;
  const seqMatch = phones.length === oracle.length && phones.every((s, i) => s[2] === oracle[i][2]);
  let maxDiff = 0;
  for (let i = 0; i < oracle.length; i++)
    maxDiff = Math.max(maxDiff, Math.abs(phones[i][0] - oracle[i][0]), Math.abs(phones[i][1] - oracle[i][1]));

  console.log(`\nbrowser phones: ${phones.map((p) => p[2]).filter((p) => p !== '[SIL]').join(' ')}`);
  console.log(`words: ${result.words.map((w) => w[2]).join(' ')}`);
  console.log(`seq match oracle: ${seqMatch ? 'OK' : 'MISMATCH'} | max boundary diff: ${maxDiff.toFixed(2)}s`);
  pass = seqMatch && maxDiff <= 0.03;
} finally {
  await browser.close();
  server.close();
}

console.log(`\n${pass ? 'PASS: in-browser (onnxruntime-web) alignment matches charsiu' : 'FAIL'}`);
process.exit(pass ? 0 : 1);

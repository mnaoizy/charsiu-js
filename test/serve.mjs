// Static server for the browser demo: `npm run serve` then open the printed URL.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { extname } from 'node:path';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const MIME = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.wav': 'audio/wav',
  '.onnx': 'application/octet-stream', '.bin': 'application/octet-stream', '.css': 'text/css' };

const server = createServer(async (req, res) => {
  try {
    let path = decodeURIComponent(req.url.split('?')[0]);
    if (path === '/') path = '/web/index.html';
    const data = await readFile(ROOT + path.replace(/^\//, ''));
    res.setHeader('Content-Type', MIME[extname(path)] || 'application/octet-stream');
    res.end(data);
  } catch { res.statusCode = 404; res.end('not found'); }
});

const port = Number(process.env.PORT) || 8080;
server.listen(port, () => console.log(`charsiu-js demo: http://localhost:${port}/web/index.html`));

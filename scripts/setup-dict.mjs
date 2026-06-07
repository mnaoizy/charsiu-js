// Build a tokana dictionary for the Japanese aligner.
//   npm run setup-dict            # IPADIC  -> models/ipadic-dict  (~12 MB, default)
//   npm run setup-dict -- neologd # NEologd  -> models/neologd-dict (large: a much
//                                   bigger lexicon — proper nouns, neologisms, names)
// NEologd = IPADIC base + the NEologd seed CSVs merged, compiled with tokana.
// tokana auto-detects each CSV's charset (IPADIC is EUC-JP, NEologd seed is UTF-8).
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const which = (process.argv[2] || 'ipadic').toLowerCase();
if (which !== 'ipadic' && which !== 'neologd') {
  console.error(`unknown dictionary "${which}" (use: ipadic | neologd)`);
  process.exit(1);
}
const root = fileURLToPath(new URL('../', import.meta.url));
const out = join(root, 'models', which === 'neologd' ? 'neologd-dict' : 'ipadic-dict');
const tokanaCli = join(root, 'node_modules', 'tokana', 'dist', 'cli.js');
const URL_IPADIC = 'https://sourceforge.net/projects/mecab/files/mecab-ipadic/2.7.0-20070801/mecab-ipadic-2.7.0-20070801.tar.gz/download';
const NEOLOGD_API = 'https://api.github.com/repos/neologd/mecab-ipadic-neologd/contents/seed';

if (existsSync(join(out, 'base.dat.gz'))) { console.log(`already built: ${out}`); process.exit(0); }
if (!existsSync(tokanaCli)) { console.error('tokana not installed — run `npm install`.'); process.exit(1); }

const run = (cmd, args) => execFileSync(cmd, args, { stdio: 'inherit' });
const tmp = mkdtempSync(join(tmpdir(), `${which}-`));
const tgz = join(tmp, 'ipadic.tar.gz');
const src = join(tmp, 'mecab-ipadic-2.7.0-20070801');

// 1. IPADIC base (matrix.def, char.def, unk.def + base CSVs) — both formats need it.
console.log('downloading mecab-ipadic (base)…');
run('curl', ['-sL', '-o', tgz, URL_IPADIC]);
run('tar', ['xzf', tgz, '-C', tmp]);

// 2. NEologd: download every seed/*.csv.xz, decompress into the IPADIC dir.
if (which === 'neologd') {
  console.log('fetching NEologd seed list…');
  const items = await fetch(NEOLOGD_API, { headers: { 'User-Agent': 'charsiu-js' } }).then((r) => r.json());
  const seeds = items.filter((i) => i.name.endsWith('.csv.xz'));
  console.log(`downloading ${seeds.length} NEologd seed files (~50 MB)…`);
  for (const s of seeds) {
    const xz = join(src, s.name);
    run('curl', ['-sL', '-o', xz, s.download_url]);
    run('xz', ['-d', '-f', xz]);            // -> src/<name>.csv (UTF-8)
  }
  console.log(`merged NEologd seeds; ${readdirSync(src).filter((f) => f.endsWith('.csv')).length} CSV files total`);
}

// 3. compile with tokana. NEologd has millions of entries -> give the build a
//    big V8 heap (the double-array trie is memory-hungry).
console.log(`compiling (${which}) -> ${out} … (NEologd has millions of entries; this can take a while)`);
const heap = which === 'neologd' ? ['--max-old-space-size=16384'] : [];
run('node', [...heap, tokanaCli, 'build', src, out, '--format', which]);
console.log(`done: ${out}`);

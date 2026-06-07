// Japanese kana -> phoneme conversion, targeting the OpenJTalk/pyopenjtalk phone
// set used by prj-beatrice/japanese-hubert-base-phoneme-ctc (a i u e o k g s z t d
// n h b p m y r w f j v ch ts sh + palatalized *y, kw gw, N cl). Input is the
// katakana *pronunciation* reading (e.g. tokana's `pronunciation` field, or
// pyopenjtalk's kana=True output): long vowels as 'ー', geminate as 'ッ', moraic
// nasal as 'ン'.
//
// Note: vowel devoicing (OpenJTalk's 'U'/'I') is context/accent dependent and not
// recoverable from kana alone, so this emits voiced 'u'/'i'. For forced alignment
// that is fine — the target is constrained and the boundary cost is barely affected.

// 2-kana digraphs (base + small kana) — matched before single kana.
const DIGRAPH: Record<string, string[]> = {
  キャ: ['ky', 'a'], キュ: ['ky', 'u'], キョ: ['ky', 'o'], キェ: ['ky', 'e'],
  ギャ: ['gy', 'a'], ギュ: ['gy', 'u'], ギョ: ['gy', 'o'],
  シャ: ['sh', 'a'], シュ: ['sh', 'u'], ショ: ['sh', 'o'], シェ: ['sh', 'e'],
  ジャ: ['j', 'a'], ジュ: ['j', 'u'], ジョ: ['j', 'o'], ジェ: ['j', 'e'],
  チャ: ['ch', 'a'], チュ: ['ch', 'u'], チョ: ['ch', 'o'], チェ: ['ch', 'e'],
  ヂャ: ['j', 'a'], ヂュ: ['j', 'u'], ヂョ: ['j', 'o'],
  ニャ: ['ny', 'a'], ニュ: ['ny', 'u'], ニョ: ['ny', 'o'],
  ヒャ: ['hy', 'a'], ヒュ: ['hy', 'u'], ヒョ: ['hy', 'o'],
  ビャ: ['by', 'a'], ビュ: ['by', 'u'], ビョ: ['by', 'o'],
  ピャ: ['py', 'a'], ピュ: ['py', 'u'], ピョ: ['py', 'o'],
  ミャ: ['my', 'a'], ミュ: ['my', 'u'], ミョ: ['my', 'o'],
  リャ: ['ry', 'a'], リュ: ['ry', 'u'], リョ: ['ry', 'o'],
  // foreign / extended
  ファ: ['f', 'a'], フィ: ['f', 'i'], フェ: ['f', 'e'], フォ: ['f', 'o'], フュ: ['fy', 'u'],
  ティ: ['t', 'i'], テュ: ['ty', 'u'], ディ: ['d', 'i'], デュ: ['dy', 'u'],
  トゥ: ['t', 'u'], ドゥ: ['d', 'u'], ニェ: ['ny', 'e'],
  ウィ: ['w', 'i'], ウェ: ['w', 'e'], ウォ: ['w', 'o'],
  ツァ: ['ts', 'a'], ツィ: ['ts', 'i'], ツェ: ['ts', 'e'], ツォ: ['ts', 'o'],
  スィ: ['s', 'i'], ズィ: ['z', 'i'],
  イェ: ['y', 'e'],
  ヴァ: ['v', 'a'], ヴィ: ['v', 'i'], ヴェ: ['v', 'e'], ヴォ: ['v', 'o'], ヴュ: ['by', 'u'],
  クァ: ['kw', 'a'], クィ: ['kw', 'i'], クェ: ['kw', 'e'], クォ: ['kw', 'o'],
  クヮ: ['kw', 'a'], グァ: ['gw', 'a'], グヮ: ['gw', 'a'],
};

// single kana -> phonemes
const MONO: Record<string, string[]> = {
  ア: ['a'], イ: ['i'], ウ: ['u'], エ: ['e'], オ: ['o'],
  カ: ['k', 'a'], キ: ['k', 'i'], ク: ['k', 'u'], ケ: ['k', 'e'], コ: ['k', 'o'],
  ガ: ['g', 'a'], ギ: ['g', 'i'], グ: ['g', 'u'], ゲ: ['g', 'e'], ゴ: ['g', 'o'],
  サ: ['s', 'a'], シ: ['sh', 'i'], ス: ['s', 'u'], セ: ['s', 'e'], ソ: ['s', 'o'],
  ザ: ['z', 'a'], ジ: ['j', 'i'], ズ: ['z', 'u'], ゼ: ['z', 'e'], ゾ: ['z', 'o'],
  タ: ['t', 'a'], チ: ['ch', 'i'], ツ: ['ts', 'u'], テ: ['t', 'e'], ト: ['t', 'o'],
  ダ: ['d', 'a'], ヂ: ['j', 'i'], ヅ: ['z', 'u'], デ: ['d', 'e'], ド: ['d', 'o'],
  ナ: ['n', 'a'], ニ: ['n', 'i'], ヌ: ['n', 'u'], ネ: ['n', 'e'], ノ: ['n', 'o'],
  ハ: ['h', 'a'], ヒ: ['h', 'i'], フ: ['f', 'u'], ヘ: ['h', 'e'], ホ: ['h', 'o'],
  バ: ['b', 'a'], ビ: ['b', 'i'], ブ: ['b', 'u'], ベ: ['b', 'e'], ボ: ['b', 'o'],
  パ: ['p', 'a'], ピ: ['p', 'i'], プ: ['p', 'u'], ペ: ['p', 'e'], ポ: ['p', 'o'],
  マ: ['m', 'a'], ミ: ['m', 'i'], ム: ['m', 'u'], メ: ['m', 'e'], モ: ['m', 'o'],
  ヤ: ['y', 'a'], ユ: ['y', 'u'], ヨ: ['y', 'o'],
  ラ: ['r', 'a'], リ: ['r', 'i'], ル: ['r', 'u'], レ: ['r', 'e'], ロ: ['r', 'o'],
  ワ: ['w', 'a'], ヰ: ['i'], ヱ: ['e'], ヲ: ['o'], ヴ: ['v', 'u'],
  // small vowels standing alone
  ァ: ['a'], ィ: ['i'], ゥ: ['u'], ェ: ['e'], ォ: ['o'],
  ャ: ['y', 'a'], ュ: ['y', 'u'], ョ: ['y', 'o'],
  ヶ: ['k', 'a'], ヵ: ['k', 'a'],
};

const VOWELS = new Set(['a', 'i', 'u', 'e', 'o']);

/** Convert a katakana pronunciation reading to a phoneme sequence. */
export function kanaToPhonemes(kana: string): string[] {
  const out: string[] = [];
  let lastVowel = '';
  const push = (ps: string[]) => {
    for (const p of ps) { out.push(p); if (VOWELS.has(p)) lastVowel = p; }
  };
  const s = kana.normalize('NFKC');
  for (let i = 0; i < s.length; ) {
    const two = s.slice(i, i + 2);
    const ch = s[i];
    if (DIGRAPH[two]) { push(DIGRAPH[two]); i += 2; continue; }
    if (ch === 'ッ') { out.push('cl'); lastVowel = ''; i += 1; continue; }
    if (ch === 'ン') { out.push('N'); lastVowel = ''; i += 1; continue; }
    if (ch === 'ー' || ch === '－' || ch === '〜' || ch === '~') {
      if (lastVowel) out.push(lastVowel);          // long vowel -> repeat
      i += 1; continue;
    }
    if (MONO[ch]) { push(MONO[ch]); i += 1; continue; }
    // skip anything else (spaces, punctuation, unknown)
    i += 1;
  }
  return out;
}

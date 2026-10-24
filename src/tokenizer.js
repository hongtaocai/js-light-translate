/**
 * SentencePiece-compatible tokenizer for the translation model.
 *
 * The browser has no sentencepiece build, so this reimplements what the model
 * needs: NFKC normalisation, the ▁ word marker, BPE encoding by merge rank,
 * UTF-8 byte fallback, and decoding.
 *
 * BPE encoding is not longest-match and not score-summing: pieces are numbered
 * in merge order, and the encoder repeatedly applies the lowest-numbered
 * adjacent merge until none applies. That is why "billionaire" segments as
 * ▁billion | a | ire even though "air" exists in the vocabulary.
 */

export class Tokenizer {
  constructor(vocab) {
    this.data = vocab;
    this.pieces = vocab.pieces;
    this.unkId = vocab.unk_id;
    this.bosId = vocab.bos_id;
    this.eosId = vocab.eos_id;
    this.padId = vocab.pad_id;

    this.ids = new Map();
    for (let i = 0; i < this.pieces.length; i += 1) {
      this.ids.set(this.pieces[i], i);
    }
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to fetch vocabulary: ${res.status}`);
    return new Tokenizer(await res.json());
  }

  /** NFKC, collapse whitespace, mark word starts with ▁, add a leading marker. */
  normalise(text) {
    let s = text.normalize('NFKC').replace(/\s+/g, ' ').trim();
    s = s.replace(/ /g, '▁');
    if (!s.startsWith('▁')) s = '▁' + s;
    return s;
  }

  byteIds(ch) {
    const out = [];
    for (const byte of new TextEncoder().encode(ch)) {
      const hex = byte.toString(16).toUpperCase().padStart(2, '0');
      const id = this.ids.get(`<0x${hex}>`);
      if (id !== undefined) out.push(id);
    }
    return out;
  }

  /** BPE: repeatedly merge the adjacent pair whose combined piece has the lowest id. */
  segment(s) {
    const symbols = Array.from(s);

    for (;;) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i + 1 < symbols.length; i += 1) {
        const id = this.ids.get(symbols[i] + symbols[i + 1]);
        if (id !== undefined && id < bestRank) {
          bestRank = id;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      symbols.splice(bestAt, 2, symbols[bestAt] + symbols[bestAt + 1]);
    }

    const ids = [];
    for (const symbol of symbols) {
      const id = this.ids.get(symbol);
      if (id !== undefined) ids.push(id);
      else ids.push(...this.byteIds(symbol));
    }
    return ids;
  }

  tokenize(text, maxTokens = 0) {
    const ids = this.segment(this.normalise(text));
    return maxTokens ? ids.slice(0, maxTokens) : ids;
  }

  /** Model input: BOS + tokens + EOS, capped at maxLen-2 tokens. */
  encode(text, maxLen = 100) {
    const ids = this.tokenize(text, maxLen - 2);
    return [this.bosId, ...ids, this.eosId];
  }

  decode(ids) {
    let out = '';
    for (const id of ids) {
      const piece = this.pieces[id];
      if (piece === undefined) continue;
      if (piece.startsWith('<0x') && piece.endsWith('>')) {
        out += String.fromCharCode(parseInt(piece.slice(3, -1), 16));
      } else {
        out += piece;
      }
    }
    return out.replace(/▁/g, ' ').trim();
  }

  /** Rough source-language guess, same 30% CJK threshold as the Python CLI. */
  static detect(text) {
    const chars = [...text].filter((c) => !/\s/.test(c));
    if (!chars.length) return 'en';
    const cjk = chars.filter((c) => {
      const cp = c.codePointAt(0);
      return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf)
        || (cp >= 0x20000 && cp <= 0x2a6df);
    }).length;
    return cjk / chars.length > 0.3 ? 'zh' : 'en';
  }
}

/**
 * Browser runtime for the translation model.
 *
 * The PyTorch checkpoint is exported to two ONNX graphs (encoder + decoder), so
 * greedy decoding happens here: run the encoder once, then call the decoder
 * once per generated token.
 *
 *     import * as ort from 'onnxruntime-web';
 *     const translator = await Translator.load({ ort });
 *     const { text } = await translator.translate('Hello world');
 */

import { Tokenizer } from './tokenizer.js';

const DEFAULTS = {
  modelBase: 'models/onnx',
  vocabUrl: 'models/vocab.json',
  quantized: true,
  maxSourceTokens: 100,
  maxNewTokens: 60,
};

export class Translator {
  constructor({ ort, tokenizer, encoder, decoder, backend }) {
    this.ort = ort;
    this.tokenizer = tokenizer;
    this.encoder = encoder;
    this.decoder = decoder;
    this.backend = backend;
    this.stats = { encodeMs: 0, decodeMs: 0, tokens: 0 };
  }

  static async load(options = {}) {
    const cfg = { ...DEFAULTS, ...options };
    const { ort } = cfg;
    if (!ort) throw new Error('Translator.load needs an onnxruntime-web namespace as `ort`');

    // WebGPU when the browser has it, WASM otherwise.
    const preferred = [];
    if (cfg.backend) preferred.push(cfg.backend);
    else if (ort.env?.webgpu && typeof navigator !== 'undefined' && navigator.gpu) preferred.push('webgpu');
    preferred.push('wasm');

    const suffix = cfg.quantized ? '.int8.onnx' : '.onnx';
    const tokenizer = await Tokenizer.load(cfg.vocabUrl);

    // Sessions MUST be created one at a time: onnxruntime-web allows only one
    // WebGPU execution provider registration at a time, and parallel creation
    // fails with "another WebGPU EP inference session is being created".
    const create = async (providers) => {
      const encoder = await ort.InferenceSession.create(`${cfg.modelBase}/encoder${suffix}`, {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      });
      const decoder = await ort.InferenceSession.create(`${cfg.modelBase}/decoder${suffix}`, {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      });
      return { encoder, decoder };
    };

    let sessions;
    let providers = preferred;
    try {
      sessions = await create(providers);
    } catch (error) {
      if (preferred[0] !== 'webgpu') throw error;
      console.warn('WebGPU session failed, retrying on WASM:', error);
      providers = ['wasm'];
      sessions = await create(providers);
    }
    const { encoder, decoder } = sessions;

    const backend = encoder.executionProviders?.[0]
      ?? (providers[0] === 'webgpu' ? 'webgpu' : 'wasm');

    return new Translator({ ort, tokenizer, encoder, decoder, backend });
  }

  int64Tensor(values, dims) {
    return new this.ort.Tensor('int64', BigInt64Array.from(values, (v) => BigInt(v)), dims);
  }

  boolTensor(values, dims) {
    return new this.ort.Tensor('bool', Uint8Array.from(values, (v) => (v ? 1 : 0)), dims);
  }

  /**
   * Translate one string. `onToken` is called with each decoded token so the UI
   * can stream the output.
   */
  async translate(text, { maxNewTokens = DEFAULTS.maxNewTokens, onToken } = {}) {
    const t0 = performance.now();
    const tokenizer = this.tokenizer;
    const sourceIds = tokenizer.encode(text, DEFAULTS.maxSourceTokens);
    const srcIds = this.int64Tensor(sourceIds, [1, sourceIds.length]);
    const srcMask = this.boolTensor(sourceIds.map((id) => id === tokenizer.padId), [1, sourceIds.length]);

    const memory = await this.encoder.run({ src_ids: srcIds, src_mask: srcMask });
    const memoryTensor = memory.memory;
    const t1 = performance.now();

    const generated = [];
    const ys = [tokenizer.bosId];

    for (let step = 0; step < maxNewTokens; step += 1) {
      const tgtIds = this.int64Tensor(ys, [1, ys.length]);
      const out = await this.decoder.run({
        tgt_ids: tgtIds,
        memory: memoryTensor,
        src_mask: srcMask,
      });

      const logits = out.logits;
      const [, timeSteps, vocab] = logits.dims;      // [1, T, V]
      const offset = (timeSteps - 1) * vocab;
      const data = logits.data;

      let best = 0;
      let bestScore = -Infinity;
      for (let v = 0; v < vocab; v += 1) {
        const score = data[offset + v];
        if (score > bestScore) {
          bestScore = score;
          best = v;
        }
      }

      if (best === tokenizer.eosId) break;
      ys.push(best);
      generated.push(best);

      const piece = tokenizer.decode([best]);
      if (onToken) onToken(piece, tokenizer.decode(generated));
    }

    const t2 = performance.now();
    this.stats = {
      encodeMs: t1 - t0,
      decodeMs: t2 - t1,
      tokens: generated.length,
      tokensPerSecond: generated.length ? (generated.length * 1000) / (t2 - t1) : 0,
    };

    return { text: tokenizer.decode(generated), tokens: generated.length, stats: this.stats };
  }

  static detect(text) {
    return Tokenizer.detect(text);
  }
}

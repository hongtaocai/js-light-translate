# Light Translate

A 22.7M-parameter English ↔ Chinese Transformer that runs **entirely in the
browser**. No server, no API calls: the weights are exported to ONNX (int8) and
executed with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).

```
index.html            demo page
src/tokenizer.js      SentencePiece-compatible tokenizer (BPE + byte fallback)
src/translator.js     ONNX sessions + greedy decoding
src/app.js            demo wiring
src/styles.css        UI
models/vocab.json     16k pieces, exported from sentencepiece
models/onnx/          encoder/decoder graphs (fp32 and int8)
models/spm.model      original sentencepiece model (for reference)
models/model.pt       source checkpoint (PyTorch)
scripts/              export + quantization utilities
```

## Run it

The page fetches model files, so it needs an HTTP origin (not `file://`):

```bash
cd /home/jcai/p/js-light-translate
python3 -m http.server 8080
# open http://localhost:8080
```

First load pulls ~36 MB (encoder 13 MB + decoder 22 MB), so give it a few
seconds on a cold cache. WebGPU is used when available; otherwise it falls back
to WASM (slower, but works everywhere).

## Library

```js
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.bundle.min.mjs';
import { Translator } from './src/translator.js';

const translator = await Translator.load({ ort });

const { text, stats } = await translator.translate('The weather is nice today.', {
  onToken: (piece, soFar) => console.log(soFar),
});
console.log(text);          // 今天天气不错。
console.log(stats.tokensPerSecond);
```

`Translator.load({ quantized: false })` uses the fp32 graphs instead of int8.

## Model

| | |
|---|---|
| Architecture | Encoder–decoder Transformer, 4 + 4 layers |
| `d_model` | 384, 6 heads, FFN 1536 |
| Vocabulary | 16,000 SentencePiece BPE pieces (shared EN/ZH) |
| Parameters | 22.7M |
| Decoding | Greedy, max 60 new tokens |
| Validation | 44.7 BLEU en→zh, 28.0 BLEU zh→en (corpus BLEU-4, 200 sentences) |

Trained on a mixed corpus (textbooks, OPUS-100, dictionary examples, fiction,
news) with the Dion3 optimizer.

## Rebuilding the ONNX files

```bash
# from the training repo's venv (needs torch + onnx)
python scripts/export_onnx.py --ckpt models/model.pt --out models/onnx
python scripts/export_vocab.py
python scripts/quantize_verify.py     # int8 + sanity check
```

`export_onnx.py` splits the model into two graphs because the browser drives the
decoding loop: the encoder runs once per sentence, the decoder once per token.

## Notes

- The tokenizer reimplements SentencePiece's normaliser (NFKC), the `▁` word
  marker, greedy longest-match encoding and UTF-8 byte fallback.
- No KV cache: every decode step recomputes the target prefix. At 60 tokens and
  128 max length that's cheap enough, and it keeps the ONNX graphs simple.
- The demo has no backend component; it will work from any static host.

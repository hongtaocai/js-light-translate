/**
 * Node-side integration test: loads the int8 graphs with onnxruntime-web's node
 * build and runs the exact Translator.translate() path the browser uses.
 */
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { Translator } from '/home/jcai/p/js-light-translate/src/translator.js';
import { Tokenizer } from '/home/jcai/p/js-light-translate/src/tokenizer.js';

const ROOT = '/home/jcai/p/js-light-translate';
const vocab = JSON.parse(readFileSync(`${ROOT}/models/vocab.json`, 'utf8'));
const tokenizer = new Tokenizer(vocab);

const encoder = await ort.InferenceSession.create(`${ROOT}/models/onnx/encoder.int8.onnx`);
const decoder = await ort.InferenceSession.create(`${ROOT}/models/onnx/decoder.int8.onnx`);

const translator = new Translator({ ort, tokenizer, encoder, decoder, backend: 'wasm' });

for (const text of [
  'The weather is nice today.',
  'He opened the letter and read it twice.',
  'California’s billionaire tax faces a mountain of opposition',
]) {
  const t0 = Date.now();
  const result = await translator.translate(text);
  console.log(`${text}\n  -> ${result.text}   (${result.tokens} tok, ${Date.now() - t0} ms)`);
}

// Chinese -> English
const zh = await translator.translate('今天天气很好');
console.log(`今天天气很好\n  -> ${zh.text}`);

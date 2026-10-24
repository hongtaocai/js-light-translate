/**
 * Demo wiring: load the model on demand, translate, stream tokens into the
 * output pane. All inference happens locally through onnxruntime-web.
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.bundle.min.mjs';
import { Translator } from './translator.js';

// ORT's WASM build routes C++ warnings (node placement, powerPreference) through
// console.error. They are informational; keep the console to real errors.
ort.env.logLevel = 'error';

const $ = (id) => document.getElementById(id);
const els = {
  load: $('load'),
  translate: $('translate'),
  input: $('input'),
  output: $('output'),
  stats: $('stats'),
  copy: $('copy'),
  clear: $('clear'),
  swap: $('swap'),
  chars: $('char-count'),
  backend: $('chip-backend'),
  hint: $('hint'),
  dirSource: $('dir-source'),
  dirTarget: $('dir-target'),
};

let translator = null;
let busy = false;

function setBackend(text, state = 'warn') {
  els.backend.className = `chip ${state}`;
  els.backend.querySelector('span').textContent = text;
}

function updateDirection() {
  const text = els.input.value.trim();
  if (!text) return;
  const source = Translator.detect(text);
  els.dirSource.textContent = source.toUpperCase();
  els.dirTarget.textContent = source === 'zh' ? 'EN' : 'ZH';
}

function updateChars() {
  const n = els.input.value.length;
  els.chars.textContent = `${n} character${n === 1 ? '' : 's'}`;
  updateDirection();
}

function setOutput(text, streaming = false) {
  els.output.innerHTML = '';
  if (!text) {
    const p = document.createElement('span');
    p.className = 'placeholder';
    p.textContent = 'Load the model, then press Translate.';
    els.output.append(p);
    return;
  }
  els.output.append(document.createTextNode(text));
  if (streaming) {
    const caret = document.createElement('span');
    caret.className = 'caret';
    els.output.append(caret);
  }
  els.output.scrollTop = els.output.scrollHeight;
}

async function loadModel() {
  if (translator || busy) return;
  busy = true;
  els.load.disabled = true;
  els.load.querySelector('.spinner').hidden = false;
  els.load.querySelector('.label').textContent = 'Loading…';
  setBackend('loading model…', 'warn');

  const started = performance.now();
  try {
    translator = await Translator.load({ ort });
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    setBackend(`${translator.backend} · ready in ${seconds}s`, 'ok');
    els.load.classList.add('ready');
    els.load.querySelector('.label').textContent = 'Model ready';
    els.translate.disabled = false;
    els.hint.innerHTML = `Running on <code>${translator.backend}</code> — inference stays on this device.`;
  } catch (error) {
    console.error(error);
    setBackend('load failed', 'warn');
    els.load.disabled = false;
    els.load.querySelector('.label').textContent = 'Retry load';
    els.hint.textContent = `Could not load the model: ${error.message}`;
  } finally {
    busy = false;
    els.load.querySelector('.spinner').hidden = true;
  }
}

async function translate() {
  if (!translator || busy) return;
  const text = els.input.value.trim();
  if (!text) return;

  busy = true;
  els.translate.disabled = true;
  els.translate.classList.add('is-busy');
  els.copy.disabled = true;
  setOutput('', false);
  els.stats.textContent = 'working…';

  let streamed = '';
  try {
    const result = await translator.translate(text, {
      onToken: (piece, sofar) => {
        streamed = sofar;
        setOutput(sofar, true);
      },
    });

    setOutput(result.text || '(empty result)', false);
    els.copy.disabled = !result.text;
    const s = result.stats;
    els.stats.textContent = `${result.tokens} tokens · ${Math.round(s.decodeMs)} ms decode · `
      + `${s.tokensPerSecond.toFixed(1)} tok/s`;
  } catch (error) {
    console.error(error);
    setOutput(`Translation failed: ${error.message}`, false);
    els.stats.textContent = '—';
  } finally {
    busy = false;
    els.translate.disabled = false;
    els.translate.classList.remove('is-busy');
  }
}

els.load.addEventListener('click', loadModel);
els.translate.addEventListener('click', translate);

els.input.addEventListener('input', updateChars);
els.input.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    event.preventDefault();
    translate();
  }
});

els.clear.addEventListener('click', () => {
  els.input.value = '';
  updateChars();
  setOutput('', false);
  els.stats.textContent = '—';
  els.copy.disabled = true;
  els.input.focus();
});

els.swap.addEventListener('click', () => {
  const text = els.input.value;
  if (!text.trim()) return;
  const swapped = els.output.textContent.trim();
  if (!swapped) return;
  els.input.value = swapped;
  updateChars();
  setOutput('', false);
});

els.copy.addEventListener('click', async () => {
  const text = els.output.textContent;
  if (!text) return;
  await navigator.clipboard.writeText(text);
  const original = els.copy.textContent;
  els.copy.textContent = 'Copied';
  setTimeout(() => { els.copy.textContent = original; }, 1200);
});

updateChars();
setOutput('', false);

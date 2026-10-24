#!/usr/bin/env python3
"""Quantise the ONNX graphs to int8 and verify greedy decoding end to end."""
import os
import sys

sys.path.insert(0, "/home/jcai/p/minitranslation")

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from model.config import ModelConfig
from model.transformer import Attention, Transformer

ROOT = "/home/jcai/p/js-light-translate"
ONNX = os.path.join(ROOT, "models", "onnx")

# the export-time patch: no isneginf, which has no ONNX kernel
import math


def _attention_forward_no_guard(self, q, k, v, mask=None, key_padding_mask=None):
    B, Lq, _ = q.shape
    Lk = k.shape[1]
    q = self.q_proj(q).view(B, Lq, self.nhead, self.d_head).transpose(1, 2)
    k = self.k_proj(k).view(B, Lk, self.nhead, self.d_head).transpose(1, 2)
    v = self.v_proj(v).view(B, Lk, self.nhead, self.d_head).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = self.dropout(attn)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, -1)
    return self.out_proj(out)


Attention.forward = _attention_forward_no_guard

for name in ("encoder", "decoder"):
    src = os.path.join(ONNX, f"{name}.onnx")
    dst = os.path.join(ONNX, f"{name}.int8.onnx")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    print(f"{name}: {os.path.getsize(src)/1e6:.0f} MB -> {os.path.getsize(dst)/1e6:.0f} MB")

# --- verify: greedy decode with the quantised graphs vs PyTorch ---
ckpt = torch.load(os.path.join(ROOT, "models", "model.pt"), map_location="cpu", weights_only=False)
cfg = ModelConfig()
for k, v in ckpt["config"].items():
    setattr(cfg, k, v)
model = Transformer(cfg)
model.load_state_dict(ckpt["model"])
model.eval()
sp = spm.SentencePieceProcessor(model_file=os.path.join(ROOT, "models", "spm.model"))

enc = ort.InferenceSession(os.path.join(ONNX, "encoder.int8.onnx"), providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(os.path.join(ONNX, "decoder.int8.onnx"), providers=["CPUExecutionProvider"])
enc32 = ort.InferenceSession(os.path.join(ONNX, "encoder.onnx"), providers=["CPUExecutionProvider"])
dec32 = ort.InferenceSession(os.path.join(ONNX, "decoder.onnx"), providers=["CPUExecutionProvider"])


def encode(text):
    ids = sp.encode(text, out_type=int)[: cfg.max_len - 2]
    return [sp.bos_id()] + ids + [sp.eos_id()]


def greedy(session_enc, session_dec, text, rounds=40):
    ids = np.array([encode(text)], dtype=np.int64)
    mask = (ids == cfg.pad_id)
    memory = session_enc.run(None, {"src_ids": ids, "src_mask": mask})[0]
    ys = [sp.bos_id()]
    for _ in range(rounds):
        tgt = np.array([ys], dtype=np.int64)
        logits = session_dec.run(None, {"tgt_ids": tgt, "memory": memory, "src_mask": mask})[0]
        nxt = int(logits[0, -1].argmax())
        ys.append(nxt)
        if nxt == sp.eos_id():
            break
    return sp.decode(ys[1:-1] if ys[-1] == sp.eos_id() else ys[1:])


for text in ["The weather is nice today.", "California’s billionaire tax faces a mountain of opposition"]:
    print("\nin :", text)
    print("fp32:", greedy(enc32, dec32, text))
    print("int8:", greedy(enc, dec, text))

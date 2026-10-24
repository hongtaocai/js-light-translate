#!/usr/bin/env python3
"""Export the transformer to ONNX for onnxruntime-web.

The browser cannot run PyTorch, so the model is split into two graphs:

  encoder.onnx : src_ids[B,L], src_mask[B,L]            -> memory[B,L,D]
  decoder.onnx : tgt_ids[B,T], memory[B,L,D], src_mask  -> logits[B,T,V]

Greedy decoding then lives in JavaScript: run the encoder once and call the
decoder once per generated token.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/jcai/p/minitranslation")

from model.config import ModelConfig
from model.transformer import Attention, Transformer


def _attention_forward_no_guard(self, q, k, v, mask=None, key_padding_mask=None):
    """Same as Attention.forward but without the all-masked-row guard.

    `torch.isneginf` has no ONNX export, and at inference every query row has at
    least one valid key (the source always starts with BOS; the causal mask
    always allows the query itself), so the guard cannot trigger here.
    """
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


class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_ids, src_mask):
        return self.model.encode(src_ids, src_mask)


class DecoderWrapper(nn.Module):
    """Full decoder pass (no KV cache): O(T^2) per step, fine for T <= 128."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tgt_ids, memory, src_mask):
        m = self.model
        x = m.embedding(tgt_ids) * math.sqrt(m.cfg.d_model)
        x = m.pos_enc(x)
        length = tgt_ids.size(1)
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=tgt_ids.device))
        for layer in m.decoder_layers:
            x = layer(x, memory, src_mask, causal, None)
        return m.lm_head(m.norm_d(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig()
    for k, v in ckpt["config"].items():
        setattr(cfg, k, v)
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"model: d_model={cfg.d_model} layers={cfg.num_encoder_layers}/{cfg.num_decoder_layers} "
          f"vocab={cfg.vocab_size}")

    os.makedirs(args.out, exist_ok=True)
    B, L, T = 1, 24, 8
    src = torch.randint(5, cfg.vocab_size, (B, L))
    src_mask = torch.zeros(B, L, dtype=torch.bool)
    tgt = torch.randint(5, cfg.vocab_size, (B, T))

    with torch.no_grad():
        memory = EncoderWrapper(model)(src, src_mask)
        logits = DecoderWrapper(model)(tgt, memory, src_mask)

    torch.onnx.export(
        EncoderWrapper(model).eval(),
        (src, src_mask),
        os.path.join(args.out, "encoder.onnx"),
        input_names=["src_ids", "src_mask"],
        output_names=["memory"],
        dynamic_axes={"src_ids": {0: "batch", 1: "src_len"},
                      "src_mask": {0: "batch", 1: "src_len"},
                      "memory": {0: "batch", 1: "src_len"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print("encoder.onnx written")

    torch.onnx.export(
        DecoderWrapper(model).eval(),
        (tgt, memory, src_mask),
        os.path.join(args.out, "decoder.onnx"),
        input_names=["tgt_ids", "memory", "src_mask"],
        output_names=["logits"],
        dynamic_axes={"tgt_ids": {0: "batch", 1: "tgt_len"},
                      "memory": {0: "batch", 1: "src_len"},
                      "src_mask": {0: "batch", 1: "src_len"},
                      "logits": {0: "batch", 1: "tgt_len"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print("decoder.onnx written")

    # Sanity: ONNX Runtime output must match PyTorch.
    import numpy as np
    import onnxruntime as ort

    enc = ort.InferenceSession(os.path.join(args.out, "encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(os.path.join(args.out, "decoder.onnx"), providers=["CPUExecutionProvider"])
    mem_onnx = enc.run(None, {"src_ids": src.numpy(), "src_mask": src_mask.numpy()})[0]
    logits_onnx = dec.run(None, {"tgt_ids": tgt.numpy(), "memory": mem_onnx,
                                 "src_mask": src_mask.numpy()})[0]
    with torch.no_grad():
        mem_ref = EncoderWrapper(model)(src, src_mask).numpy()
        log_ref = DecoderWrapper(model)(tgt, torch.from_numpy(mem_ref), src_mask).numpy()
    print(f"max |encoder diff| = {np.abs(mem_onnx - mem_ref).max():.2e}")
    print(f"max |decoder diff| = {np.abs(logits_onnx - log_ref).max():.2e}")


if __name__ == "__main__":
    main()

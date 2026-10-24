#!/usr/bin/env python3
"""Dump the SentencePiece vocabulary as JSON for the browser tokenizer."""
import json
import os

import sentencepiece as spm

ROOT = "/home/jcai/p/js-light-translate"
sp = spm.SentencePieceProcessor(model_file=os.path.join(ROOT, "models", "spm.model"))

pieces, scores = [], []
for i in range(sp.get_piece_size()):
    pieces.append(sp.id_to_piece(i))
    scores.append(round(sp.get_score(i), 4))

data = {
    "pieces": pieces,
    "scores": scores,
    "unk_id": sp.unk_id(),
    "bos_id": sp.bos_id(),
    "eos_id": sp.eos_id(),
    "pad_id": sp.pad_id(),
    "max_piece_len": max(len(p) for p in pieces),
}
out = os.path.join(ROOT, "models", "vocab.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

print(f"pieces: {len(pieces)}, max length: {data['max_piece_len']}, "
      f"{os.path.getsize(out)/1e6:.1f} MB")
print("sample:", pieces[:6])
print("byte pieces:", sum(1 for p in pieces if p.startswith("<0x")))

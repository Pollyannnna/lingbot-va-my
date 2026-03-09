#!/usr/bin/env python3
"""Create LingBotVA-compatible empty_emb.pt from tokenizer + text_encoder."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import T5TokenizerFast, UMT5EncoderModel


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate empty_emb.pt for LingBotVA post-training.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Model root containing tokenizer/ and text_encoder/ directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path to save empty_emb.pt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt text used for embedding (default empty string).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length for tokenizer padding/truncation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=tuple(DTYPE_MAP.keys()),
        help="Saved embedding dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for text encoding: auto/cpu/cuda/cuda:N",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_model_root(model_root: Path) -> tuple[Path, Path]:
    tokenizer_path = model_root / "tokenizer"
    text_encoder_path = model_root / "text_encoder"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer dir not found: {tokenizer_path}")
    if not text_encoder_path.exists():
        raise FileNotFoundError(f"text_encoder dir not found: {text_encoder_path}")
    return tokenizer_path, text_encoder_path


def main() -> int:
    args = _parse_args()
    model_root = args.model_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        print(f"output exists, use --overwrite: {output_path}")
        return 2

    tokenizer_path, text_encoder_path = _validate_model_root(model_root)
    save_dtype = DTYPE_MAP[args.dtype]
    device = _resolve_device(args.device)

    tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_path))
    # Keep encoder in fp32 for numerical stability, cast only final tensor.
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(text_encoder_path),
        torch_dtype=torch.float32,
    ).to(device)
    text_encoder.eval()

    inputs = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=args.max_seq_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)
    seq_len = int(attn_mask[0].sum().item())

    with torch.no_grad():
        hidden = text_encoder(input_ids, attn_mask).last_hidden_state[0]  # [L, D]

    # Follow LingBot prompt embedding behavior: valid tokens then zero padding to max_seq_len.
    valid = hidden[:seq_len]
    if seq_len < args.max_seq_len:
        pad = torch.zeros(
            args.max_seq_len - seq_len,
            hidden.shape[1],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        emb = torch.cat([valid, pad], dim=0)
    else:
        emb = valid

    emb = emb.to("cpu", dtype=save_dtype).contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, output_path)

    # Save-and-reload validation.
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, torch.Tensor):
        print(f"saved object is not Tensor: {type(loaded)}")
        return 1
    if loaded.ndim != 2:
        print(f"saved tensor rank must be 2, got shape={tuple(loaded.shape)}")
        return 1
    if loaded.shape[0] != args.max_seq_len:
        print(f"saved tensor first dim must be max_seq_len={args.max_seq_len}, got {loaded.shape[0]}")
        return 1

    print(f"saved: {output_path}")
    print(f"shape: {tuple(loaded.shape)} dtype: {loaded.dtype} device: cpu")
    print(f"non_zero_rows: {(loaded.abs().sum(dim=1) > 0).sum().item()} / {loaded.shape[0]}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())

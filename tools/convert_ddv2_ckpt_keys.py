#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert DiffusionDriveV2 / SparseDriveV2 checkpoint keys to the layout expected
by the current DiffusionDriveV2Policy wrapper.

Typical mismatch:
    checkpoint key:
        _backbone.image_encoder.conv1.weight
        _trajectory_head.diff_decoder...
        bev_proj.0.weight

    model expected key:
        _transfuser_model._backbone.image_encoder.conv1.weight
        _transfuser_model._trajectory_head.diff_decoder...
        _transfuser_model.bev_proj.0.weight

Usage:
    python tools/convert_ddv2_ckpt_keys.py \
        --input DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt \
        --output DiffusionDriveV2/ckpt/diffusiondrivev2_rl_compatible.ckpt

    python tools/convert_ddv2_ckpt_keys.py \
        --input outputs/actor_learner/weights/latest.ckpt \
        --output outputs/actor_learner/weights/latest_compatible.ckpt
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def torch_load(path: str | Path) -> Any:
    """Compatible torch.load for different PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values())


def find_state_dict_slot(ckpt: Any) -> tuple[str | None, dict[str, Any]]:
    """
    Return:
        slot_key = None means ckpt itself is state_dict.
        slot_key = "state_dict" means ckpt["state_dict"] is state_dict.
    """
    if looks_like_state_dict(ckpt):
        return None, ckpt

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "agent",
        "policy",
        "net",
        "module",
    ]

    for key in candidate_keys:
        value = ckpt.get(key, None)
        if looks_like_state_dict(value):
            return key, value

    for key, value in ckpt.items():
        if looks_like_state_dict(value):
            return key, value

    raise RuntimeError(
        "Cannot find a tensor state_dict in checkpoint. "
        f"Top-level keys: {list(ckpt.keys())[:30]}"
    )


def strip_training_prefixes(key: str) -> str:
    """
    Remove common training-wrapper prefixes.

    Examples:
        agent._backbone.xxx -> _backbone.xxx
        module.agent._backbone.xxx -> _backbone.xxx
        policy._trajectory_head.xxx -> _trajectory_head.xxx
    """
    prefixes = (
        "module.",
        "model.",
        "agent.",
        "policy.",
        "actor.",
        "learner.",
        "_model.",
    )

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break

    return key


def normalize_ddv2_key(key: str) -> str:
    """
    Convert raw DiffusionDriveV2 keys to wrapper keys.

    Raw / official-like:
        _backbone.xxx
        _trajectory_head.xxx
        _agent_head.xxx
        _tf_decoder.xxx
        bev_proj.xxx

    Wrapper expected:
        _transfuser_model._backbone.xxx
        _transfuser_model._trajectory_head.xxx
        _transfuser_model._agent_head.xxx
        _transfuser_model._tf_decoder.xxx
        _transfuser_model.bev_proj.xxx
    """
    original_key = str(key)
    key = strip_training_prefixes(original_key)

    # Already compatible.
    if key.startswith("_transfuser_model."):
        return key

    transfuser_prefixes = (
        "_backbone.",
        "_keyval_embedding.",
        "_query_embedding.",
        "_bev_downscale.",
        "_status_encoding.",
        "_bev_semantic_head.",
        "_tf_decoder.",
        "_agent_head.",
        "_trajectory_head.",
        "bev_proj.",
    )

    if key.startswith(transfuser_prefixes):
        return "_transfuser_model." + key

    # Keep unrelated keys unchanged, e.g. optimizer, scheduler, global_step metadata.
    return key


def summarize_prefixes(keys: list[str], topk: int = 20) -> list[tuple[str, int]]:
    def prefix_of(k: str) -> str:
        parts = k.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
        return parts[0]

    return Counter(prefix_of(k) for k in keys).most_common(topk)


def convert_checkpoint(input_path: Path, output_path: Path, dry_run: bool = False) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    ckpt = torch_load(input_path)
    slot_key, state_dict = find_state_dict_slot(ckpt)

    old_keys = list(state_dict.keys())
    new_state_dict: dict[str, Any] = {}

    renamed = 0
    collisions: list[tuple[str, str]] = []
    examples: list[tuple[str, str]] = []

    for old_key, value in state_dict.items():
        new_key = normalize_ddv2_key(old_key)

        if new_key != old_key:
            renamed += 1
            if len(examples) < 20:
                examples.append((old_key, new_key))

        if new_key in new_state_dict:
            collisions.append((old_key, new_key))

        new_state_dict[new_key] = value

    if collisions:
        print("[WARNING] Key collisions detected after conversion:")
        for old_key, new_key in collisions[:20]:
            print(f"  {old_key} -> {new_key}")
        raise RuntimeError(f"Aborted because {len(collisions)} key collisions were detected.")

    new_keys = list(new_state_dict.keys())

    print("=" * 80)
    print(f"Input ckpt:  {input_path}")
    print(f"Output ckpt: {output_path}")
    print(f"State dict slot: {slot_key if slot_key is not None else '<top-level state_dict>'}")
    print(f"Total keys:   {len(old_keys)}")
    print(f"Renamed keys: {renamed}")
    print("-" * 80)
    print("Old key prefix summary:")
    for prefix, count in summarize_prefixes(old_keys):
        print(f"  {prefix:<45} {count}")
    print("-" * 80)
    print("New key prefix summary:")
    for prefix, count in summarize_prefixes(new_keys):
        print(f"  {prefix:<45} {count}")
    print("-" * 80)
    print("Examples:")
    if examples:
        for old_key, new_key in examples:
            print(f"  {old_key}")
            print(f"    -> {new_key}")
    else:
        print("  No key was renamed. The checkpoint may already be compatible.")
    print("=" * 80)

    if dry_run:
        print("[dry-run] No file was saved.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if slot_key is None:
        new_ckpt = new_state_dict
    else:
        # Preserve optimizer / scheduler / epoch / global_step and other metadata.
        new_ckpt = dict(ckpt)
        new_ckpt[slot_key] = new_state_dict

    torch.save(new_ckpt, output_path)
    print(f"[OK] Saved compatible checkpoint to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DiffusionDriveV2 checkpoint keys to _transfuser_model.* layout."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the original checkpoint.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Path to save the converted checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print conversion summary; do not save checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if input_path == output_path:
        raise RuntimeError("Input and output paths must be different.")

    convert_checkpoint(
        input_path=input_path,
        output_path=output_path,
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Hold out a fixed Video-o3 subset for RL validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


VIDEOO3_ANNOTATIONS = "video-o3_with_grounded_cot_7891.json"


def split_videoo3(rows: list[dict], val_size: int = 100, seed: int = 42) -> tuple[list[dict], list[dict]]:
    if not 0 < val_size < len(rows):
        raise ValueError(f"--val-size must be between 1 and {len(rows) - 1}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:val_size])
    train = [row for index, row in enumerate(rows) if index not in val_indices]
    val = [row for index, row in enumerate(rows) if index in val_indices]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.data_root / "RL" / VIDEOO3_ANNOTATIONS
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {source}")
    train, val = split_videoo3(rows, args.val_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "videoo3_train.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "videoo3_val.json").write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
    print(f"Video-o3: {len(train)} train, {len(val)} validation (seed={args.seed})")


if __name__ == "__main__":
    main()

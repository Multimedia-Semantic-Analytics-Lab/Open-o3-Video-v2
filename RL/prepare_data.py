"""Create a deterministic, disjoint train/validation split for STGR-v2 RL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be between zero and one")
    source = args.data_root / "RL" / "STGR-RL-Refine-0501-17615.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"Expected a JSON list with at least two rows: {source}")
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    val_size = max(1, round(len(rows) * args.val_ratio))
    val_indices = set(indices[:val_size])
    train = [row for index, row in enumerate(rows) if index not in val_indices]
    val = [row for index, row in enumerate(rows) if index in val_indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rl_train.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "rl_val.json").write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
    print(f"STGR-v2 RL: {len(train)} train, {len(val)} validation")


if __name__ == "__main__":
    main()

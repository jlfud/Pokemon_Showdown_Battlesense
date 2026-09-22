"""Parse raw replay JSONL and create leakage-safe train/validation/test files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from src.data.parse_logs import parse_battle


def split_for_battle(battle_id: str, seed: int = 42) -> str:
    """Assign a battle deterministically: 70% train, 15% val, 15% test."""
    digest = hashlib.sha256(f"{seed}:{battle_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "val"
    return "test"


def read_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    battle_counts: Counter[str] = Counter()
    example_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    skipped = 0

    try:
        records = read_jsonl(input_path, limit=limit)
        for record in tqdm(records, total=limit, desc="Parsing battles"):
            if record.get("formatid") != "gen8ou":
                skipped += 1
                continue
            examples = parse_battle(record)
            if not examples:
                skipped += 1
                continue
            split = split_for_battle(str(record.get("id", "unknown")), seed=seed)
            battle_counts[split] += 1
            for example in examples:
                handles[split].write(json.dumps(example, separators=(",", ":")) + "\n")
                example_counts[split] += 1
                label_counts[f"{split}_{example['label_action_type']}"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    metadata = {
        "source": str(input_path),
        "format_id": "gen8ou",
        "seed": seed,
        "battle_counts": dict(battle_counts),
        "example_counts": dict(example_counts),
        "label_counts": dict(label_counts),
        "skipped_battles": skipped,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/gen8ou_battles.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = prepare_dataset(args.input, args.output_dir, args.limit, args.seed)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

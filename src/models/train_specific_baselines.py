"""Train species-aware frequency baselines for specific moves and switches.

Pokemon Showdown replay logs expose move names, but not a stable four-slot order
for an opponent's hidden moveset.  This model therefore predicts move names
directly and serves as the Phase 2 benchmark for future learned predictors.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MATCHUP_SEPARATOR = "\t"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _serializable(nested: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(counts) for key, counts in nested.items()}


def train(input_path: Path, output_path: Path) -> dict[str, Any]:
    global_counts = {"move": Counter(), "switch": Counter()}
    species_counts: dict[str, defaultdict[str, Counter[str]]] = {
        "move": defaultdict(Counter),
        "switch": defaultdict(Counter),
    }
    matchup_counts: dict[str, defaultdict[str, Counter[str]]] = {
        "move": defaultdict(Counter),
        "switch": defaultdict(Counter),
    }
    examples = Counter()

    for row in _read_jsonl(input_path):
        action_type = row["label_action_type"]
        if action_type not in global_counts:
            continue
        target = row["label_target"]
        active = row["active_species"]
        opponent = row["opponent_species"]
        matchup = f"{active}{MATCHUP_SEPARATOR}{opponent}"
        global_counts[action_type][target] += 1
        species_counts[action_type][active][target] += 1
        matchup_counts[action_type][matchup][target] += 1
        examples[action_type] += 1

    artifact = {
        "model_type": "specific_action_frequency_baseline",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_examples": dict(examples),
        "global_counts": {
            action: dict(counts) for action, counts in global_counts.items()
        },
        "species_counts": {
            action: _serializable(counts) for action, counts in species_counts.items()
        },
        "matchup_counts": {
            action: _serializable(counts) for action, counts in matchup_counts.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, separators=(",", ":"), sort_keys=True)
    return {
        "output": str(output_path),
        "training_examples": dict(examples),
        "unique_moves": len(global_counts["move"]),
        "unique_switch_targets": len(global_counts["switch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/v1_specific_action_baseline.json"),
    )
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

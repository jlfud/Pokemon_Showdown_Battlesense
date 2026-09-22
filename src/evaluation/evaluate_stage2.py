"""Evaluate combined top-k action predictions on the validation split."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.model_manager import ModelManager
from src.models.train_stage1 import IGNORED_COLUMNS


def _turn_bin(turn: int) -> str:
    if turn <= 10:
        return "1-10"
    if turn <= 20:
        return "11-20"
    if turn <= 30:
        return "21-30"
    return "31+"


def _rate(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def evaluate(
    data_path: Path,
    model_dir: Path,
    output_path: Path,
    errors_path: Path,
    error_sample_size: int = 200,
) -> dict[str, Any]:
    frame = pd.read_json(data_path, lines=True)
    manager = ModelManager(model_dir)
    model_frame = frame.drop(
        columns=[column for column in IGNORED_COLUMNS if column in frame],
        errors="ignore",
    )
    action_probabilities = manager.action_model.predict_proba(
        manager.preprocessor.transform(model_frame)
    )

    totals = Counter()
    correct_at_1 = Counter()
    correct_at_3 = Counter()
    conditional_at_1 = Counter()
    conditional_at_3 = Counter()
    turn_totals = Counter()
    turn_correct_1 = Counter()
    turn_correct_3 = Counter()
    error_sample: list[dict[str, Any]] = []
    seen_errors = 0
    rng = random.Random(42)

    for row, probabilities in zip(frame.to_dict("records"), action_probabilities):
        action_type = row["label_action_type"]
        actual = (action_type, row["label_target"])
        predictions = manager.predict(
            row,
            top_k=3,
            action_probabilities=(float(probabilities[0]), float(probabilities[1])),
        )
        predicted = [
            (prediction["action_type"], prediction["target"])
            for prediction in predictions
        ]
        totals["overall"] += 1
        totals[action_type] += 1
        if predicted and actual == predicted[0]:
            correct_at_1["overall"] += 1
            correct_at_1[action_type] += 1
        if actual in predicted:
            correct_at_3["overall"] += 1
            correct_at_3[action_type] += 1

        candidates = row.get("available_switches") if action_type == "switch" else None
        conditional = manager.specific_model.distribution(
            action_type,
            row["active_species"],
            row["opponent_species"],
            candidates=candidates,
        )
        conditional_ranked = sorted(
            conditional, key=conditional.get, reverse=True
        )
        if conditional_ranked and row["label_target"] == conditional_ranked[0]:
            conditional_at_1[action_type] += 1
        if row["label_target"] in conditional_ranked[:3]:
            conditional_at_3[action_type] += 1

        turn_group = _turn_bin(int(row["turn_number"]))
        turn_totals[turn_group] += 1
        if predicted and actual == predicted[0]:
            turn_correct_1[turn_group] += 1
        if actual in predicted:
            turn_correct_3[turn_group] += 1

        if not predicted or actual != predicted[0]:
            seen_errors += 1
            error = {
                "battle_id": row["battle_id"],
                "turn_number": row["turn_number"],
                "active_species": row["active_species"],
                "opponent_species": row["opponent_species"],
                "available_switches": row.get("available_switches", []),
                "actual": {"action_type": actual[0], "target": actual[1]},
                "predictions": predictions,
            }
            if len(error_sample) < error_sample_size:
                error_sample.append(error)
            else:
                index = rng.randrange(seen_errors)
                if index < error_sample_size:
                    error_sample[index] = error

    metrics = {
        "dataset": str(data_path),
        "examples": totals["overall"],
        "test_set_evaluated": False,
        "combined": {
            "top_1_accuracy": _rate(correct_at_1["overall"], totals["overall"]),
            "top_3_accuracy": _rate(correct_at_3["overall"], totals["overall"]),
        },
        "by_action_type": {
            action: {
                "examples": totals[action],
                "combined_top_1": _rate(correct_at_1[action], totals[action]),
                "combined_top_3": _rate(correct_at_3[action], totals[action]),
                "conditional_top_1": _rate(conditional_at_1[action], totals[action]),
                "conditional_top_3": _rate(conditional_at_3[action], totals[action]),
            }
            for action in ("move", "switch")
        },
        "by_turn": {
            group: {
                "examples": turn_totals[group],
                "top_1_accuracy": _rate(turn_correct_1[group], turn_totals[group]),
                "top_3_accuracy": _rate(turn_correct_3[group], turn_totals[group]),
            }
            for group in ("1-10", "11-20", "21-30", "31+")
        },
        "error_sample_size": len(error_sample),
        "notes": [
            "Specific-action predictions are a frequency baseline, not the final Stage 2 classifiers.",
            "The test split remains untouched.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8") as handle:
        for error in error_sample:
            handle.write(json.dumps(error, separators=(",", ":")) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/metrics/stage2_validation.json")
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=Path("results/error_samples/stage2_validation_errors.jsonl"),
    )
    args = parser.parse_args()
    metrics = evaluate(args.data, args.model_dir, args.output, args.errors)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

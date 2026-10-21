"""Train and evaluate the candidate-scoring action ranker.

Each decision is expanded into one row per candidate action (see
``src.models.candidates``).  Frequency features for training rows come from
out-of-fold tables, grouped by battle, so a row's own label never feeds its
features.  Training decisions whose actual action is not among the candidates
are dropped; on validation they count as misses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from xgboost import XGBClassifier, XGBRanker

from src.models.candidates import (
    DEFAULT_MOVE_CANDIDATES,
    FEATURE_NAMES,
    FrequencyTables,
    candidate_actions,
    candidate_features,
)


FOLDS = 5
TURN_BINS = ("1-10", "11-20", "21-30", "31+")


def fold_for_battle(battle_id: str, folds: int = FOLDS) -> int:
    digest = hashlib.sha256(f"fold:{battle_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _turn_bin(turn: int) -> str:
    if turn <= 10:
        return "1-10"
    if turn <= 20:
        return "11-20"
    if turn <= 30:
        return "21-30"
    return "31+"


def build_tables(train_path: Path) -> tuple[FrequencyTables, list[FrequencyTables]]:
    """Full training tables plus, for each fold, tables excluding that fold."""
    per_fold = [FrequencyTables() for _ in range(FOLDS)]
    for row in _read_jsonl(train_path):
        per_fold[fold_for_battle(row["battle_id"])].add(row)
    full = FrequencyTables()
    for tables in per_fold:
        full += tables
    return full, [full - tables for tables in per_fold]


def build_matrix(
    path: Path,
    tables_for: Callable[[dict[str, Any]], FrequencyTables],
    move_k: int,
    drop_uncovered: bool,
) -> dict[str, Any]:
    """Candidate feature rows for every decision in ``path``."""
    blocks: list[np.ndarray] = []
    group_sizes: list[int] = []
    labels: list[int] = []
    decisions: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        tables = tables_for(row)
        actions = candidate_actions(row, tables, move_k)
        actual = (row["label_action_type"], row["label_target"])
        label = actions.index(actual) if actual in actions else -1
        if not actions or (drop_uncovered and label < 0):
            continue
        blocks.append(np.asarray(candidate_features(row, actions, tables), dtype=np.float32))
        group_sizes.append(len(actions))
        labels.append(label)
        decisions.append(
            {
                "action_type": row["label_action_type"],
                "turn_bin": _turn_bin(int(row["turn_number"])),
                "actions": actions,
            }
        )
    relevance = np.zeros(sum(group_sizes), dtype=np.float32)
    starts = np.concatenate([[0], np.cumsum(group_sizes)[:-1]]).astype(int)
    for start, label in zip(starts, labels):
        if label >= 0:
            relevance[start + label] = 1.0
    return {
        "x": np.concatenate(blocks),
        "relevance": relevance,
        "qid": np.repeat(np.arange(len(group_sizes)), group_sizes),
        "starts": starts,
        "sizes": np.asarray(group_sizes),
        "labels": np.asarray(labels),
        "decisions": decisions,
    }


def group_probabilities(
    margins: np.ndarray, starts: np.ndarray, sizes: np.ndarray, temperature: float
) -> list[np.ndarray]:
    probabilities = []
    for start, size in zip(starts, sizes):
        scores = margins[start : start + size] / temperature
        scores = np.exp(scores - scores.max())
        probabilities.append(scores / scores.sum())
    return probabilities


def _covered_nll(probabilities: list[np.ndarray], labels: np.ndarray) -> float:
    losses = [-math.log(max(p[label], 1e-12)) for p, label in zip(probabilities, labels) if label >= 0]
    return float(np.mean(losses))


def fit_temperature(margins: np.ndarray, data: dict[str, Any]) -> float:
    """Softmax temperature minimizing log-loss on covered decisions."""
    candidates = np.exp(np.linspace(math.log(0.1), math.log(10.0), 61))
    losses = [
        _covered_nll(
            group_probabilities(margins, data["starts"], data["sizes"], t), data["labels"]
        )
        for t in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def evaluate(probabilities: list[np.ndarray], data: dict[str, Any]) -> dict[str, Any]:
    totals, top1, top3, covered = Counter(), Counter(), Counter(), Counter()
    for p, label, decision in zip(probabilities, data["labels"], data["decisions"]):
        ranked = np.argsort(-p)
        for key in ("overall", decision["action_type"], decision["turn_bin"]):
            totals[key] += 1
            if label >= 0:
                covered[key] += 1
                top1[key] += int(ranked[0] == label)
                top3[key] += int(label in ranked[:3])

    def summary(key: str) -> dict[str, Any]:
        n = totals[key]
        return {
            "examples": n,
            "top_1_accuracy": top1[key] / n if n else 0.0,
            "top_3_accuracy": top3[key] / n if n else 0.0,
            "candidate_coverage": covered[key] / n if n else 0.0,
        }

    return {
        "combined": {
            **summary("overall"),
            "log_loss_covered": _covered_nll(probabilities, data["labels"]),
        },
        "by_action_type": {action: summary(action) for action in ("move", "switch")},
        "by_turn": {group: summary(group) for group in TURN_BINS},
    }


def make_model(objective: str) -> XGBRanker | XGBClassifier:
    params = dict(
        n_estimators=2000,
        learning_rate=0.08,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
    )
    if objective == "binary":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
    return XGBRanker(
        objective=f"rank:{objective}",
        eval_metric="ndcg@1",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=1,
        **params,
    )


def train(
    data_dir: Path,
    model_dir: Path,
    metrics_path: Path,
    objective: str,
    move_k: int,
) -> dict[str, Any]:
    full_tables, oof_tables = build_tables(data_dir / "train.jsonl")
    train_data = build_matrix(
        data_dir / "train.jsonl",
        lambda row: oof_tables[fold_for_battle(row["battle_id"])],
        move_k,
        drop_uncovered=True,
    )
    val_data = build_matrix(
        data_dir / "val.jsonl", lambda row: full_tables, move_k, drop_uncovered=False
    )

    model = make_model(objective)
    fit_args: dict[str, Any] = {"verbose": False}
    if objective != "binary":
        fit_args["qid"] = train_data["qid"]
        fit_args["eval_qid"] = [val_data["qid"]]
    model.fit(
        train_data["x"],
        train_data["relevance"],
        eval_set=[(val_data["x"], val_data["relevance"])],
        **fit_args,
    )

    margins = model.predict(val_data["x"], output_margin=True)
    temperature = fit_temperature(margins, val_data)
    probabilities = group_probabilities(
        margins, val_data["starts"], val_data["sizes"], temperature
    )
    importance = model.get_booster().get_score(importance_type="gain")
    named_importance = {
        FEATURE_NAMES[int(key[1:])]: value for key, value in importance.items()
    }

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_dir / "val.jsonl"),
        "test_set_evaluated": False,
        "objective": objective,
        "move_candidates_per_species": move_k,
        "best_iteration": int(model.best_iteration),
        "temperature": temperature,
        "train_decisions": len(train_data["sizes"]),
        "train_candidate_rows": int(train_data["x"].shape[0]),
        **evaluate(probabilities, val_data),
        "top_features_by_gain": [
            [name, gain]
            for name, gain in sorted(named_importance.items(), key=lambda item: -item[1])[:15]
        ],
        "notes": [
            "Temperature and early stopping use the validation split; the test split remains untouched.",
            "log_loss_covered excludes decisions whose actual action is not a candidate.",
        ],
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "v2_ranker_model.json")
    with (model_dir / "v2_ranker_tables.json").open("w", encoding="utf-8") as handle:
        json.dump(full_tables.to_dict(), handle, separators=(",", ":"))
    with (model_dir / "v2_ranker_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {**metrics, "feature_names": FEATURE_NAMES}, handle, indent=2, sort_keys=True
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--metrics", type=Path, default=Path("results/metrics/ranker_validation.json")
    )
    parser.add_argument("--objective", choices=("ndcg", "pairwise", "binary"), default="binary")
    parser.add_argument("--move-candidates", type=int, default=DEFAULT_MOVE_CANDIDATES)
    args = parser.parse_args()
    metrics = train(
        args.data_dir, args.model_dir, args.metrics, args.objective, args.move_candidates
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

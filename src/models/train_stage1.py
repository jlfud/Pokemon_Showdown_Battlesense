"""Train the Phase 1 move-versus-switch classifier."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


IGNORED_COLUMNS = {
    "battle_id",
    "format_id",
    "actor_player",
    "actor_side",
    "actor_rating",
    "battle_rating",
    "label_action_type",
    "label_action_type_id",
    "label_target",
    "team_species",
    "opponent_team_species",
    "available_switches",
    "active_revealed_moves",
    "opponent_revealed_moves",
    "bench_hp",
    "bench_status",
}


def load_jsonl(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    if frame.empty:
        raise ValueError(f"No examples found in {path}")
    return frame


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidate_columns = [column for column in frame.columns if column not in IGNORED_COLUMNS]
    numeric = [column for column in candidate_columns if is_numeric_dtype(frame[column])]
    categorical = [column for column in candidate_columns if column not in numeric]
    return numeric, categorical


def evaluate(y_true: pd.Series, y_pred: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["move", "switch"],
            output_dict=True,
            zero_division=0,
        ),
    }


def train(data_dir: Path, model_dir: Path) -> dict:
    train_frame = load_jsonl(data_dir / "train.jsonl")
    val_frame = load_jsonl(data_dir / "val.jsonl")
    numeric, categorical = feature_columns(train_frame)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=3),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )

    x_train = preprocessor.fit_transform(train_frame)
    x_val = preprocessor.transform(val_frame)
    y_train = train_frame["label_action_type_id"].astype(int)
    y_val = val_frame["label_action_type_id"].astype(int)

    move_count = int((y_train == 0).sum())
    switch_count = int((y_train == 1).sum())
    if switch_count == 0:
        raise ValueError("Training set contains no switch examples")
    # A fully balanced weight over-predicts switches and hurts the README's
    # primary accuracy metric.  Cap the correction while still improving
    # minority-class recall over an unweighted model.
    scale_pos_weight = min(1.5, move_count / switch_count)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_predictions = model.predict(x_val)
    majority_class = int(y_train.mode().iloc[0])
    majority_predictions = [majority_class] * len(y_val)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "task": "gen8ou_action_type",
        "train_examples": len(train_frame),
        "validation_examples": len(val_frame),
        "train_move_examples": move_count,
        "train_switch_examples": switch_count,
        "scale_pos_weight": scale_pos_weight,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "majority_baseline": evaluate(y_val, majority_predictions),
        "xgboost": evaluate(y_val, val_predictions),
        "note": "The test split was intentionally not evaluated during development.",
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "v1_action_type_model.json")
    joblib.dump(preprocessor, model_dir / "v1_preprocessor.joblib")
    with (model_dir / "v1_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.model_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

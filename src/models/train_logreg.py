"""Train logistic regression classifier for move-versus-switch prediction."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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


def train_logreg(data_dir: Path, model_dir: Path) -> dict:
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

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        solver="saga",
    )
    model.fit(x_train, y_train)

    val_predictions = model.predict(x_val)
    majority_class = int(y_train.mode().iloc[0])
    majority_predictions = [majority_class] * len(y_val)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "task": "gen8ou_action_type",
        "model_type": "logistic_regression",
        "train_examples": len(train_frame),
        "validation_examples": len(val_frame),
        "train_move_examples": move_count,
        "train_switch_examples": switch_count,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "majority_baseline": evaluate(y_val, majority_predictions),
        "logreg": evaluate(y_val, val_predictions),
        "note": "The test split was intentionally not evaluated during development.",
    }

    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model and preprocessor together as a pipeline for consistent loading
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    joblib.dump(full_pipeline, model_dir / "v1_logreg_model.joblib")

    with (model_dir / "v1_logreg_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    print(json.dumps(train_logreg(args.data_dir, args.model_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

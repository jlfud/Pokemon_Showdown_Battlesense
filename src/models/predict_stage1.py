"""Run move-versus-switch inference on a processed BattleSense example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.models.train_stage1 import IGNORED_COLUMNS


def predict(example: dict, model_dir: Path) -> dict:
    preprocessor = joblib.load(model_dir / "v1_preprocessor.joblib")
    model = XGBClassifier()
    model.load_model(model_dir / "v1_action_type_model.json")
    frame = pd.DataFrame([example]).drop(
        columns=[column for column in IGNORED_COLUMNS if column in example],
        errors="ignore",
    )
    probabilities = model.predict_proba(preprocessor.transform(frame))[0]
    return {
        "move_probability": float(probabilities[0]),
        "switch_probability": float(probabilities[1]),
        "prediction": "switch" if probabilities[1] >= probabilities[0] else "move",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=Path, help="JSON file containing one processed example")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    try:
        with args.example.open(encoding="utf-8") as handle:
            example = json.load(handle)
    except json.JSONDecodeError as error:
        parser.error(f"{args.example} is not valid JSON: {error}")
    try:
        result = predict(example, args.model_dir)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

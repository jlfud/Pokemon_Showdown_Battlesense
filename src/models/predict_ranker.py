"""Rank the predicted side's possible actions for one battle state with the v2 ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier, XGBRanker

from src.models.candidates import FrequencyTables, candidate_actions, candidate_features
from src.models.train_ranker import group_probabilities


# Optional fields default to "nothing revealed yet"; everything else is required.
OPTIONAL_DEFAULTS = {
    "active_last_move": "none",
    "active_revealed_moves": [],
    "opponent_revealed_moves": [],
    "bench_hp": {},
    "bench_status": {},
}


class RankerPredictor:
    def __init__(self, model_dir: Path = Path("models")):
        with (model_dir / "v2_ranker_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.temperature = metadata["temperature"]
        self.move_k = metadata["move_candidates_per_species"]
        self.model = XGBClassifier() if metadata["objective"] == "binary" else XGBRanker()
        self.model.load_model(model_dir / "v2_ranker_model.json")
        with (model_dir / "v2_ranker_tables.json").open(encoding="utf-8") as handle:
            self.tables = FrequencyTables.from_dict(json.load(handle))

    def predict(self, state: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
        state = {**OPTIONAL_DEFAULTS, **state}
        actions = candidate_actions(state, self.tables, self.move_k)
        if not actions:
            return []
        features = np.asarray(candidate_features(state, actions, self.tables), dtype=np.float32)
        margins = self.model.predict(features, output_margin=True)
        (probabilities,) = group_probabilities(
            margins, np.array([0]), np.array([len(actions)]), self.temperature
        )
        ranked = np.argsort(-probabilities)[:top_k]
        return [
            {
                "rank": rank,
                "action_type": actions[index][0],
                "target": actions[index][1],
                "probability": float(probabilities[index]),
            }
            for rank, index in enumerate(ranked, start=1)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    try:
        with args.state.open(encoding="utf-8") as handle:
            state = json.load(handle)
    except json.JSONDecodeError as error:
        parser.error(f"{args.state} is not valid JSON: {error}")
    try:
        predictions = RankerPredictor(args.model_dir).predict(state, top_k=args.top_k)
    except KeyError as error:
        parser.error(f"State is missing required field {error}")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()

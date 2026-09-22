"""Combine Stage 1 action-type and specific-action probabilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.models.train_specific_baselines import MATCHUP_SEPARATOR
from src.models.train_stage1 import IGNORED_COLUMNS


def _probabilities(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    return {target: count / total for target, count in counts.items()}


class SpecificActionBaseline:
    """Hierarchical matchup -> species -> global frequency predictor."""

    def __init__(self, artifact: dict[str, Any]):
        self.global_counts = {
            action: Counter(counts)
            for action, counts in artifact["global_counts"].items()
        }
        self.species_counts = {
            action: {species: Counter(counts) for species, counts in groups.items()}
            for action, groups in artifact["species_counts"].items()
        }
        self.matchup_counts = {
            action: {matchup: Counter(counts) for matchup, counts in groups.items()}
            for action, groups in artifact["matchup_counts"].items()
        }

    @classmethod
    def load(cls, path: Path) -> "SpecificActionBaseline":
        with path.open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def distribution(
        self,
        action_type: str,
        active_species: str,
        opponent_species: str,
        candidates: Iterable[str] | None = None,
    ) -> dict[str, float]:
        global_counts = self.global_counts[action_type]
        species_counts = self.species_counts[action_type].get(active_species, Counter())
        matchup_key = f"{active_species}{MATCHUP_SEPARATOR}{opponent_species}"
        matchup_counts = self.matchup_counts[action_type].get(matchup_key, Counter())

        if candidates is None:
            targets = set(species_counts or global_counts)
        else:
            targets = {target for target in candidates if target}
        if not targets:
            return {}

        global_prob = _probabilities(global_counts)
        species_total = sum(species_counts.values())
        species_prob = {
            target: (species_counts[target] + 5.0 * global_prob.get(target, 0.0))
            / (species_total + 5.0)
            for target in targets
        }
        matchup_total = sum(matchup_counts.values())
        if matchup_total >= 20:
            scores = {
                target: (matchup_counts[target] + 5.0 * species_prob.get(target, 0.0))
                / (matchup_total + 5.0)
                for target in targets
            }
        else:
            scores = species_prob

        score_total = sum(scores.values())
        if score_total <= 0:
            uniform = 1.0 / len(targets)
            return {target: uniform for target in targets}
        return {target: score / score_total for target, score in scores.items()}


class ModelManager:
    def __init__(self, model_dir: Path = Path("models")):
        self.preprocessor = joblib.load(model_dir / "v1_preprocessor.joblib")
        self.action_model = XGBClassifier()
        self.action_model.load_model(model_dir / "v1_action_type_model.json")
        self.specific_model = SpecificActionBaseline.load(
            model_dir / "v1_specific_action_baseline.json"
        )

    def action_type_probabilities(self, state: dict[str, Any]) -> tuple[float, float]:
        required = [
            column
            for name, _transformer, columns in self.preprocessor.transformers_
            if name != "remainder" and isinstance(columns, (list, tuple))
            for column in columns
        ]
        missing = [column for column in required if column not in state]
        if missing:
            raise ValueError(f"State is missing required fields: {', '.join(missing)}")
        frame = pd.DataFrame([state]).drop(
            columns=[column for column in IGNORED_COLUMNS if column in state],
            errors="ignore",
        )
        probabilities = self.action_model.predict_proba(
            self.preprocessor.transform(frame)
        )[0]
        return float(probabilities[0]), float(probabilities[1])

    def predict(
        self,
        state: dict[str, Any],
        top_k: int = 3,
        action_probabilities: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        move_probability, switch_probability = (
            action_probabilities
            if action_probabilities is not None
            else self.action_type_probabilities(state)
        )
        if "available_switches" not in state:
            raise ValueError(
                "State is missing available_switches; provide the predicted "
                "side's legal bench Pokemon, or [] if none are available"
            )
        available_switches = state.get("available_switches", [])
        if not available_switches:
            move_probability = 1.0
            switch_probability = 0.0

        context = {
            "active_species": state["active_species"],
            "opponent_species": state["opponent_species"],
        }
        move_distribution = self.specific_model.distribution("move", **context)
        switch_distribution = self.specific_model.distribution(
            "switch", **context, candidates=available_switches
        )

        predictions = [
            {
                "action_type": "move",
                "target": target,
                "probability": move_probability * conditional,
                "conditional_probability": conditional,
            }
            for target, conditional in move_distribution.items()
        ]
        predictions.extend(
            {
                "action_type": "switch",
                "target": target,
                "probability": switch_probability * conditional,
                "conditional_probability": conditional,
            }
            for target, conditional in switch_distribution.items()
        )
        predictions.sort(key=lambda prediction: prediction["probability"], reverse=True)
        return [
            {"rank": rank, **prediction}
            for rank, prediction in enumerate(predictions[:top_k], start=1)
        ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        with args.state.open(encoding="utf-8") as handle:
            state = json.load(handle)
    except json.JSONDecodeError as error:
        parser.error(f"{args.state} is not valid JSON: {error}")
    try:
        predictions = ModelManager(args.model_dir).predict(state, top_k=args.top_k)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()

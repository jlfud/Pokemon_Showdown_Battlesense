"""Candidate actions and per-candidate features for the action ranker.

Every decision becomes a small group of rows, one per legal or plausible action:
the predicted side's revealed and commonly used moves, plus every living bench
Pokemon.  A ranking model scores the rows and a softmax over each group gives a
single distribution over moves and switches.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from src.data.game_data import best_effectiveness, effectiveness, move_info, species_types, to_id


SMOOTHING = 5.0
DEFAULT_MOVE_CANDIDATES = 10
HAZARD_FEATURES = {
    "stealthrock": "stealth_rock",
    "spikes": "spikes",
    "toxicspikes": "toxic_spikes",
    "stickyweb": "sticky_web",
}
MISSING = math.nan

SHARED_FEATURES = [
    "turn_number",
    "active_hp",
    "opponent_hp",
    "active_has_status",
    "opponent_has_status",
    "active_boost_total",
    "opponent_boost_total",
    "active_spe_boost",
    "opponent_spe_boost",
    "team_remaining",
    "opponent_remaining",
    "own_hazard_layers",
    "opponent_hazard_layers",
    "own_stealth_rock",
    "active_just_switched_in",
    "active_revealed_move_count",
    "opponent_revealed_move_count",
    "species_switch_rate",
    "active_threatened",
    "active_offense",
]
MOVE_FEATURES = [
    "move_is_last",
    "move_is_revealed",
    "move_p_species",
    "move_p_matchup",
    "move_known",
    "move_effectiveness",
    "move_base_power",
    "move_power_score",
    "move_stab",
    "move_physical",
    "move_special",
    "move_status",
    "move_priority",
    "move_setup",
    "move_hazard",
    "move_hazard_already_set",
    "move_hazard_removal",
    "move_recovery",
    "move_pivot",
    "move_protect",
    "move_inflicts_status",
]
SWITCH_FEATURES = [
    "switch_p_species",
    "switch_p_vs_opponent",
    "switch_hp",
    "switch_has_status",
    "switch_threatened",
    "switch_offense",
    "switch_rock_weakness",
]
FEATURE_NAMES = ["is_switch", *SHARED_FEATURES, *MOVE_FEATURES, *SWITCH_FEATURES]


def _smoothed(count: int, total: int, prior: float) -> float:
    return (count + SMOOTHING * prior) / (total + SMOOTHING)


class FrequencyTables:
    """Action counts from training battles, used for candidates and priors."""

    def __init__(self) -> None:
        self.move_global: Counter[str] = Counter()
        self.move_species: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.move_matchup: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.switch_global: Counter[str] = Counter()
        self.switch_species: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.switch_vs_opponent: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.action_species: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._totals: dict[tuple[str, str], int] = {}

    _TABLES = (
        "move_species",
        "move_matchup",
        "switch_species",
        "switch_vs_opponent",
        "action_species",
    )

    def add(self, row: dict[str, Any]) -> None:
        species = row["active_species"]
        opponent = row["opponent_species"]
        target = row["label_target"]
        action_type = row["label_action_type"]
        self._totals.clear()
        self.action_species[species][action_type] += 1
        if action_type == "move":
            self.move_global[target] += 1
            self.move_species[species][target] += 1
            self.move_matchup[f"{species}\t{opponent}"][target] += 1
        else:
            self.switch_global[target] += 1
            self.switch_species[species][target] += 1
            self.switch_vs_opponent[opponent][target] += 1

    def __sub__(self, other: "FrequencyTables") -> "FrequencyTables":
        """Counts in ``self`` but not ``other``; used for out-of-fold tables."""
        result = FrequencyTables()
        result.move_global = self.move_global - other.move_global
        result.switch_global = self.switch_global - other.switch_global
        for name in self._TABLES:
            mine, theirs = getattr(self, name), getattr(other, name)
            table = getattr(result, name)
            for key, counts in mine.items():
                remaining = counts - theirs.get(key, Counter())
                if remaining:
                    table[key] = remaining
        return result

    def __iadd__(self, other: "FrequencyTables") -> "FrequencyTables":
        self._totals.clear()
        self.move_global.update(other.move_global)
        self.switch_global.update(other.switch_global)
        for name in self._TABLES:
            table = getattr(self, name)
            for key, counts in getattr(other, name).items():
                table[key].update(counts)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "move_global": dict(self.move_global),
            "switch_global": dict(self.switch_global),
            **{
                name: {key: dict(counts) for key, counts in getattr(self, name).items()}
                for name in self._TABLES
            },
        }

    @classmethod
    def from_dict(cls, artifact: dict[str, Any]) -> "FrequencyTables":
        tables = cls()
        tables.move_global = Counter(artifact["move_global"])
        tables.switch_global = Counter(artifact["switch_global"])
        for name in cls._TABLES:
            table = getattr(tables, name)
            for key, counts in artifact[name].items():
                table[key] = Counter(counts)
        return tables

    def _total(self, name: str, key: str = "") -> int:
        """Cached row total of one counter; lookups happen millions of times."""
        cache_key = (name, key)
        if cache_key not in self._totals:
            table = getattr(self, name)
            counts = table.get(key, Counter()) if key else table
            self._totals[cache_key] = sum(counts.values())
        return self._totals[cache_key]

    def top_moves(self, species: str, k: int) -> list[str]:
        return [move for move, _count in self.move_species.get(species, Counter()).most_common(k)]

    def move_probabilities(self, species: str, opponent: str, move: str) -> tuple[float, float]:
        p_global = self.move_global[move] / (self._total("move_global") or 1)
        by_species = self.move_species.get(species, Counter())
        p_species = _smoothed(by_species[move], self._total("move_species", species), p_global)
        matchup = f"{species}\t{opponent}"
        by_matchup = self.move_matchup.get(matchup, Counter())
        p_matchup = _smoothed(by_matchup[move], self._total("move_matchup", matchup), p_species)
        return p_species, p_matchup

    def switch_probabilities(self, species: str, opponent: str, target: str) -> tuple[float, float]:
        p_global = self.switch_global[target] / (self._total("switch_global") or 1)
        by_species = self.switch_species.get(species, Counter())
        by_opponent = self.switch_vs_opponent.get(opponent, Counter())
        return (
            _smoothed(by_species[target], self._total("switch_species", species), p_global),
            _smoothed(
                by_opponent[target], self._total("switch_vs_opponent", opponent), p_global
            ),
        )

    def switch_rate(self, species: str) -> float:
        counts = self.action_species.get(species, Counter())
        overall_switches = self._total("switch_global")
        overall = overall_switches + self._total("move_global")
        prior = overall_switches / overall if overall else 0.25
        return _smoothed(counts["switch"], self._total("action_species", species), prior)


def candidate_actions(
    row: dict[str, Any], tables: FrequencyTables, move_k: int = DEFAULT_MOVE_CANDIDATES
) -> list[tuple[str, str]]:
    """Moves the active Pokemon may use and the Pokemon the side may switch to.

    Once four moves are revealed the full moveset is known, so only those
    remain as move candidates.
    """
    revealed = list(row["active_revealed_moves"])
    moves = revealed if len(revealed) >= 4 else [
        *revealed,
        *tables.top_moves(row["active_species"], move_k),
    ]
    actions = [("move", move) for move in dict.fromkeys(moves)]
    actions.extend(("switch", species) for species in row["available_switches"])
    return actions


def _attacking_types(moves: Iterable[str]) -> list[str]:
    types = []
    for move in moves:
        info = move_info(move)
        if info and info["category"] != "Status":
            types.append(info["type"])
    return types


def _hazard_layers(row: dict[str, Any], prefix: str) -> float:
    return float(sum(row[f"{prefix}_{name}"] for name in HAZARD_FEATURES.values()))


def shared_features(row: dict[str, Any], tables: FrequencyTables) -> dict[str, Any]:
    """Features that are the same for every candidate of one decision, plus the
    type information the candidate features reuse."""
    active_types = species_types(row["active_species"])
    opponent_types = species_types(row["opponent_species"])
    # What the opponent's active Pokemon threatens: its own types plus any
    # attacking moves it has revealed.
    opponent_attacks = [*opponent_types, *_attacking_types(row["opponent_revealed_moves"])]
    active_attacks = [*active_types, *_attacking_types(row["active_revealed_moves"])]
    boosts = ("atk", "def", "spa", "spd", "spe")
    values = {
        "turn_number": float(row["turn_number"]),
        "active_hp": float(row["active_hp"]),
        "opponent_hp": float(row["opponent_hp"]),
        "active_has_status": float(row["active_status"] != "none"),
        "opponent_has_status": float(row["opponent_status"] != "none"),
        "active_boost_total": float(sum(row[f"active_{stat}_boost"] for stat in boosts)),
        "opponent_boost_total": float(sum(row[f"opponent_{stat}_boost"] for stat in boosts)),
        "active_spe_boost": float(row["active_spe_boost"]),
        "opponent_spe_boost": float(row["opponent_spe_boost"]),
        "team_remaining": float(row["team_remaining"]),
        "opponent_remaining": float(row["opponent_remaining"]),
        "own_hazard_layers": _hazard_layers(row, "own"),
        "opponent_hazard_layers": _hazard_layers(row, "opponent"),
        "own_stealth_rock": float(row["own_stealth_rock"]),
        "active_just_switched_in": float(row["active_last_move"] == "none"),
        "active_revealed_move_count": float(len(row["active_revealed_moves"])),
        "opponent_revealed_move_count": float(len(row["opponent_revealed_moves"])),
        "species_switch_rate": tables.switch_rate(row["active_species"]),
        "active_threatened": best_effectiveness(opponent_attacks, active_types),
        "active_offense": best_effectiveness(active_attacks, opponent_types),
    }
    return {
        "values": [values[name] for name in SHARED_FEATURES],
        "active_types": active_types,
        "opponent_types": opponent_types,
        "opponent_attacks": opponent_attacks,
    }


def _move_features(
    row: dict[str, Any], move: str, tables: FrequencyTables, context: dict[str, Any]
) -> list[float]:
    p_species, p_matchup = tables.move_probabilities(
        row["active_species"], row["opponent_species"], move
    )
    info = move_info(move)
    is_last = float(move == row["active_last_move"])
    is_revealed = float(move in row["active_revealed_moves"])
    if info is None:
        return [is_last, is_revealed, p_species, p_matchup, 0.0, *[MISSING] * 16]
    eff = effectiveness(info["type"], context["opponent_types"])
    stab = 1.5 if info["type"] in context["active_types"] else 1.0
    hazard = HAZARD_FEATURES.get(to_id(move))
    return [
        is_last,
        is_revealed,
        p_species,
        p_matchup,
        1.0,
        eff,
        info["base_power"],
        info["base_power"] * stab * eff,
        float(stab > 1.0),
        float(info["category"] == "Physical"),
        float(info["category"] == "Special"),
        float(info["category"] == "Status"),
        info["priority"],
        float(info["is_setup"]),
        float(info["is_hazard"]),
        float(row[f"opponent_{hazard}"]) if hazard else 0.0,
        float(info["is_hazard_removal"]),
        float(info["is_recovery"]),
        float(info["is_pivot"]),
        float(info["is_protect"]),
        float(info["inflicts_status"]),
    ]


def _switch_features(
    row: dict[str, Any], target: str, tables: FrequencyTables, context: dict[str, Any]
) -> list[float]:
    p_species, p_vs_opponent = tables.switch_probabilities(
        row["active_species"], row["opponent_species"], target
    )
    target_types = species_types(target)
    return [
        p_species,
        p_vs_opponent,
        float(row["bench_hp"].get(target, 1.0)),
        float(row["bench_status"].get(target, "none") != "none"),
        best_effectiveness(context["opponent_attacks"], target_types),
        best_effectiveness(target_types, context["opponent_types"]),
        effectiveness("ROCK", target_types) * row["own_stealth_rock"],
    ]


def candidate_features(
    row: dict[str, Any], actions: list[tuple[str, str]], tables: FrequencyTables
) -> list[list[float]]:
    """One feature vector per candidate, in ``FEATURE_NAMES`` order."""
    context = shared_features(row, tables)
    no_move = [MISSING] * len(MOVE_FEATURES)
    no_switch = [MISSING] * len(SWITCH_FEATURES)
    vectors = []
    for kind, target in actions:
        if kind == "move":
            specific = [*_move_features(row, target, tables, context), *no_switch]
        else:
            specific = [*no_move, *_switch_features(row, target, tables, context)]
        vectors.append([float(kind == "switch"), *context["values"], *specific])
    return vectors

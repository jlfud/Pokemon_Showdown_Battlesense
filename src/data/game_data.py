"""Gen 8 species, move and type-chart lookups backed by poke-env's game data."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable

from poke_env.data import GenData


HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
HAZARD_REMOVAL_MOVES = {"rapidspin", "defog", "courtchange"}


def to_id(name: str) -> str:
    """Showdown's ID format: lowercase alphanumerics only ("Urshifu-*" -> "urshifu")."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@lru_cache(maxsize=1)
def _data() -> GenData:
    return GenData.from_gen(8)


@lru_cache(maxsize=None)
def species_types(species: str) -> tuple[str, ...]:
    """Upper-case types for a species, or () when the species is unknown."""
    entry = _data().pokedex.get(to_id(species))
    return tuple(t.upper() for t in entry["types"]) if entry else ()


@lru_cache(maxsize=None)
def move_info(move: str) -> dict[str, Any] | None:
    """Static move properties used as ranking features, or None if unknown."""
    entry = _data().moves.get(to_id(move))
    if entry is None:
        return None
    move_id = to_id(move)
    self_boosts = entry.get("boosts") if entry.get("target") == "self" else None
    self_boosts = self_boosts or (entry.get("self") or {}).get("boosts")
    return {
        "type": entry["type"].upper(),
        "base_power": float(entry.get("basePower") or 0),
        "category": entry.get("category", "Status"),
        "priority": float(entry.get("priority") or 0),
        "is_setup": bool(self_boosts) and entry.get("category") == "Status",
        "is_hazard": move_id in HAZARD_MOVES,
        "is_hazard_removal": move_id in HAZARD_REMOVAL_MOVES,
        "is_recovery": bool(entry.get("heal")) or move_id in {"rest", "wish", "strengthsap"},
        "is_pivot": bool(entry.get("selfSwitch")),
        "is_protect": bool(entry.get("stallingMove")),
        "inflicts_status": bool(entry.get("status")),
    }


def effectiveness(attack_type: str, defender_types: Iterable[str]) -> float:
    """Damage multiplier of an attacking type; 1.0 when the defender is unknown."""
    chart = _data().type_chart
    multiplier = 1.0
    for defender in defender_types:
        multiplier *= chart.get(defender, {}).get(attack_type, 1.0)
    return multiplier


def best_effectiveness(attack_types: Iterable[str], defender_types: Iterable[str]) -> float:
    """Highest multiplier among several attacking types (1.0 if none are given)."""
    defender_types = tuple(defender_types)
    multipliers = [effectiveness(t, defender_types) for t in attack_types]
    return max(multipliers, default=1.0)

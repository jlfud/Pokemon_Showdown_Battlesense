"""Parse Pokemon Showdown singles logs into turn-level action examples.

The parser creates at most one example per side per turn.  Features are copied
from the state at the ``|turn|`` marker, before either player's choice is
revealed.  Forced replacements and pivot-move switches are state updates, not
training labels.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


SIDES = ("p1", "p2")
BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
HAZARD_LIMITS = {
    "stealth_rock": 1,
    "spikes": 3,
    "toxic_spikes": 2,
    "sticky_web": 1,
}


@dataclass
class PokemonState:
    species: str = "unknown"
    nickname: str = "unknown"
    hp: float = -1.0
    status: str = "none"
    boosts: dict[str, int] = field(
        default_factory=lambda: {stat: 0 for stat in BOOST_STATS}
    )


@dataclass
class BattleState:
    active: dict[str, PokemonState] = field(
        default_factory=lambda: {side: PokemonState() for side in SIDES}
    )
    remaining: dict[str, int] = field(default_factory=lambda: {side: 6 for side in SIDES})
    team_size: dict[str, int] = field(default_factory=lambda: {side: 6 for side in SIDES})
    preview: dict[str, list[str]] = field(
        default_factory=lambda: {side: [] for side in SIDES}
    )
    players: dict[str, str] = field(
        default_factory=lambda: {"p1": "unknown", "p2": "unknown"}
    )
    player_ratings: dict[str, float] = field(
        default_factory=lambda: {"p1": -1.0, "p2": -1.0}
    )
    weather: str = "none"
    terrain: str = "none"
    hazards: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            side: {hazard: 0 for hazard in HAZARD_LIMITS} for side in SIDES
        }
    )
    fainted: set[str] = field(default_factory=set)
    fainted_species: dict[str, set[str]] = field(
        default_factory=lambda: {side: set() for side in SIDES}
    )
    needs_replacement: set[str] = field(default_factory=set)


def _side_from_ident(value: str) -> str | None:
    side = value.strip()[:2]
    return side if side in SIDES else None


def _nickname_from_ident(value: str) -> str:
    return value.split(":", 1)[1].strip() if ":" in value else value.strip()


def _species_from_details(details: str) -> str:
    species = details.split(",", 1)[0].strip()
    return species or "unknown"


def _canonical_team_species(species: str, preview: list[str]) -> str:
    """Map revealed or transformed battle formes back to a team-preview entry."""
    if species in preview:
        return species
    for candidate in preview:
        base = candidate.removesuffix("-*")
        if candidate.endswith("-*") and (
            species == base or species.startswith(f"{base}-")
        ):
            return candidate
        if species.startswith(f"{candidate}-") and species.split("-")[-1] in {
            "Busted",
            "Noice",
        }:
            return candidate
    return species


def _condition(condition: str) -> tuple[float, str, bool]:
    """Return HP fraction, status and fainted flag from a condition string."""
    parts = condition.strip().split()
    hp = -1.0
    status = "none"
    fainted = "fnt" in parts
    if parts:
        hp_part = parts[0]
        if "/" in hp_part:
            numerator, denominator = hp_part.split("/", 1)
            try:
                hp = float(numerator) / float(denominator)
            except (ValueError, ZeroDivisionError):
                hp = -1.0
        elif hp_part == "0":
            hp = 0.0
    for token in parts[1:]:
        if token not in {"fnt"}:
            status = token
            break
    return hp, status, fainted


def _slug(effect: str) -> str:
    value = effect.lower().strip()
    for prefix in ("move: ", "ability: "):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.replace(" ", "").replace("-", "")


def _hazard_name(effect: str) -> str | None:
    normalized = _slug(effect)
    return {
        "stealthrock": "stealth_rock",
        "spikes": "spikes",
        "toxicspikes": "toxic_spikes",
        "stickyweb": "sticky_web",
    }.get(normalized)


def _terrain_name(effect: str) -> str | None:
    normalized = _slug(effect)
    for terrain in ("electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"):
        if normalized == terrain:
            return terrain.removesuffix("terrain")
    return None


def _perspective_features(snapshot: BattleState, actor: str) -> dict[str, Any]:
    opponent = "p2" if actor == "p1" else "p1"
    own = snapshot.active[actor]
    foe = snapshot.active[opponent]
    features: dict[str, Any] = {
        "actor_side": actor,
        "actor_player": snapshot.players[actor],
        "actor_rating": snapshot.player_ratings[actor],
        "active_species": own.species,
        "active_hp": own.hp,
        "active_status": own.status,
        "opponent_species": foe.species,
        "opponent_hp": foe.hp,
        "opponent_status": foe.status,
        "team_remaining": snapshot.remaining[actor],
        "opponent_remaining": snapshot.remaining[opponent],
        "weather": snapshot.weather,
        "terrain": snapshot.terrain,
        "team_species": list(snapshot.preview[actor]),
        "opponent_team_species": list(snapshot.preview[opponent]),
        "available_switches": [
            species
            for species in snapshot.preview[actor]
            if species != own.species and species not in snapshot.fainted_species[actor]
        ],
    }
    for stat in BOOST_STATS:
        features[f"active_{stat}_boost"] = own.boosts[stat]
        features[f"opponent_{stat}_boost"] = foe.boosts[stat]
    for hazard in HAZARD_LIMITS:
        features[f"own_{hazard}"] = snapshot.hazards[actor][hazard]
        features[f"opponent_{hazard}"] = snapshot.hazards[opponent][hazard]
    return features


def parse_battle(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one Hugging Face replay record into labeled action examples."""
    log = record.get("log")
    if not isinstance(log, str) or not log:
        return []

    state = BattleState()
    examples: list[dict[str, Any]] = []
    current_turn: int | None = None
    turn_start: BattleState | None = None
    acted: set[str] = set()

    def add_example(side: str, action_type: str, target: str) -> None:
        if current_turn is None or turn_start is None or side in acted:
            return
        example = {
            "battle_id": str(record.get("id", "unknown")),
            "format_id": str(record.get("formatid", "unknown")),
            "battle_rating": float(record.get("rating") or -1.0),
            "turn_number": current_turn,
            **_perspective_features(turn_start, side),
            "label_action_type": action_type,
            "label_action_type_id": 1 if action_type == "switch" else 0,
            "label_target": target,
        }
        examples.append(example)
        acted.add(side)

    for raw_line in log.splitlines():
        if not raw_line.startswith("|"):
            continue
        parts = raw_line.split("|")[1:]
        if not parts:
            continue
        event = parts[0]

        if event == "player" and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            if side:
                state.players[side] = parts[2] or "unknown"
                if len(parts) >= 5:
                    try:
                        state.player_ratings[side] = float(parts[4])
                    except ValueError:
                        pass
        elif event == "teamsize" and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            if side:
                try:
                    state.team_size[side] = int(parts[2])
                    state.remaining[side] = int(parts[2])
                except ValueError:
                    pass
        elif event == "poke" and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            if side:
                state.preview[side].append(_species_from_details(parts[2]))
                state.team_size[side] = max(state.team_size[side], len(state.preview[side]))
                state.remaining[side] = state.team_size[side]
        elif event == "turn" and len(parts) >= 2:
            try:
                current_turn = int(parts[1])
            except ValueError:
                current_turn = None
            acted = set()
            turn_start = deepcopy(state) if current_turn is not None else None
        elif event in {"switch", "drag", "replace"} and len(parts) >= 4:
            side = _side_from_ident(parts[1])
            if not side:
                continue
            species = _species_from_details(parts[2])
            is_pivot = any(value.startswith("[from]") for value in parts[4:])
            is_forced = (
                event != "switch"
                or side in state.needs_replacement
                or is_pivot
                or current_turn is None
            )
            if not is_forced:
                add_example(
                    side,
                    "switch",
                    _canonical_team_species(species, state.preview[side]),
                )
            hp, status, fainted = _condition(parts[3])
            state.active[side] = PokemonState(
                species=species,
                nickname=_nickname_from_ident(parts[1]),
                hp=hp,
                status=status,
            )
            state.needs_replacement.discard(side)
            if fainted:
                state.needs_replacement.add(side)
        elif event == "move" and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            if side and side not in state.needs_replacement:
                add_example(side, "move", parts[2])
        elif event in {"-damage", "-heal", "-sethp"} and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            if side:
                hp, status, fainted = _condition(parts[2])
                if hp >= 0:
                    state.active[side].hp = hp
                if status != "none":
                    state.active[side].status = status
                if fainted:
                    state.needs_replacement.add(side)
        elif event == "-status" and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            if side:
                state.active[side].status = parts[2]
        elif event == "-curestatus" and len(parts) >= 2:
            side = _side_from_ident(parts[1])
            if side:
                state.active[side].status = "none"
        elif event in {"-boost", "-unboost", "-setboost"} and len(parts) >= 4:
            side = _side_from_ident(parts[1])
            stat = parts[2]
            if side and stat in BOOST_STATS:
                try:
                    amount = int(parts[3])
                except ValueError:
                    continue
                if event == "-setboost":
                    value = amount
                else:
                    delta = amount if event == "-boost" else -amount
                    value = state.active[side].boosts[stat] + delta
                state.active[side].boosts[stat] = min(6, max(-6, value))
        elif event == "-clearboost" and len(parts) >= 2:
            side = _side_from_ident(parts[1])
            if side:
                state.active[side].boosts = {stat: 0 for stat in BOOST_STATS}
        elif event == "-clearallboost":
            for side in SIDES:
                state.active[side].boosts = {stat: 0 for stat in BOOST_STATS}
        elif event == "faint" and len(parts) >= 2:
            side = _side_from_ident(parts[1])
            identity = parts[1].strip()
            if side and identity not in state.fainted:
                state.fainted.add(identity)
                state.fainted_species[side].add(state.active[side].species)
                state.remaining[side] = max(0, state.remaining[side] - 1)
                state.active[side].hp = 0.0
                state.needs_replacement.add(side)
        elif event == "-weather" and len(parts) >= 2:
            weather = _slug(parts[1])
            if weather not in {"", "none"}:
                state.weather = weather
            elif not any(value == "[upkeep]" for value in parts[2:]):
                state.weather = "none"
        elif event == "-fieldstart" and len(parts) >= 2:
            terrain = _terrain_name(parts[1])
            if terrain:
                state.terrain = terrain
        elif event == "-fieldend" and len(parts) >= 2:
            if _terrain_name(parts[1]):
                state.terrain = "none"
        elif event == "-sidestart" and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            hazard = _hazard_name(parts[2])
            if side and hazard:
                state.hazards[side][hazard] = min(
                    HAZARD_LIMITS[hazard], state.hazards[side][hazard] + 1
                )
        elif event == "-sideend" and len(parts) >= 3:
            side = _side_from_ident(parts[1])
            hazard = _hazard_name(parts[2])
            if side and hazard:
                state.hazards[side][hazard] = 0

    return examples

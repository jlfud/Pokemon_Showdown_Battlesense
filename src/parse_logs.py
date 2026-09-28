from poke_env.battle import Battle
import logging
import json
import numpy as np
from poke_env.battle import Battle
from poke_env.battle.side_condition import SideCondition
from poke_env.data import GenData

logger = logging.getLogger("poke_env_replay_parser")
logger.setLevel(logging.CRITICAL)

GEN = 8
STATUS_MAP = {"": 0, "brn": 1, "par": 2, "slp": 3, "frz": 4, "psn": 5, "tox": 6}
WEATHER_MAP = {
    "none": 0,
    "sandstorm": 1,
    "sunnyday": 2,
    "desolateland": 2,
    "raindance": 3,
    "primordialsea": 3,
    "hail": 4,
    "snow": 4,
}

from poke_env.battle import Battle
from poke_env.battle.side_condition import SideCondition

def extract_side_condition_features(battle: Battle) -> list[float]:
    """
    Extracts normalized hazard and field screen states for both sides.
    Spikes range from 0 to 3 layers (normalized / 3.0).
    Toxic Spikes range from 0 to 2 layers (normalized / 2.0).
    Binary conditions (Stealth Rock, Web, Screens, Tailwind) map to 0.0 or 1.0.
    """
    p1_sides = battle.side_conditions
    p2_sides = battle.opponent_side_conditions

    return [
        # --- Player 1 Hazards & Screens ---
        1.0 if SideCondition.STEALTH_ROCK in p1_sides else 0.0,
        p1_sides.get(SideCondition.SPIKES, 0) / 3.0,
        p1_sides.get(SideCondition.TOXIC_SPIKES, 0) / 2.0,
        1.0 if SideCondition.STICKY_WEB in p1_sides else 0.0,
        1.0 if SideCondition.REFLECT in p1_sides else 0.0,
        1.0 if SideCondition.LIGHT_SCREEN in p1_sides else 0.0,
        1.0 if SideCondition.AURORA_VEIL in p1_sides else 0.0,
        1.0 if SideCondition.TAILWIND in p1_sides else 0.0,

        # --- Opponent Hazards & Screens ---
        1.0 if SideCondition.STEALTH_ROCK in p2_sides else 0.0,
        p2_sides.get(SideCondition.SPIKES, 0) / 3.0,
        p2_sides.get(SideCondition.TOXIC_SPIKES, 0) / 2.0,
        1.0 if SideCondition.STICKY_WEB in p2_sides else 0.0,
        1.0 if SideCondition.REFLECT in p2_sides else 0.0,
        1.0 if SideCondition.LIGHT_SCREEN in p2_sides else 0.0,
        1.0 if SideCondition.AURORA_VEIL in p2_sides else 0.0,
        1.0 if SideCondition.TAILWIND in p2_sides else 0.0,
    ]

def extract_state_vector(battle: Battle) -> np.ndarray:
    """Extracts a normalized, fixed-length feature vector from Player 1's perspective."""
    TOTAL_FEATURES = 30

    p1_active = battle.active_pokemon
    p2_active = battle.opponent_active_pokemon

    if not p1_active or not p2_active:
        return np.zeros(TOTAL_FEATURES, dtype=np.float32)

    p1_sides = battle.side_conditions
    p2_sides = battle.opponent_side_conditions

    features = [
        # --- Active Pokémon HP ratios (0.0 to 1.0) ---
        p1_active.current_hp_fraction if p1_active.current_hp_fraction is not None else 0.0,
        p2_active.current_hp_fraction if p2_active.current_hp_fraction is not None else 0.0,

        # --- Status conditions (mapped to scalar/categorical) ---
        STATUS_MAP.get(str(p1_active.status.name).lower() if p1_active.status else "", 0) / 6.0,
        STATUS_MAP.get(str(p2_active.status.name).lower() if p2_active.status else "", 0) / 6.0,

        # --- Stat boosts (-6 to +6 normalized to [-1, 1]) ---
        p1_active.boosts.get("atk", 0) / 6.0,
        p1_active.boosts.get("def", 0) / 6.0,
        p1_active.boosts.get("spa", 0) / 6.0,
        p1_active.boosts.get("spd", 0) / 6.0,
        p1_active.boosts.get("spe", 0) / 6.0,
        p2_active.boosts.get("atk", 0) / 6.0,
        p2_active.boosts.get("spe", 0) / 6.0,

        # --- Field state ---
        WEATHER_MAP.get(str(battle.weather).lower() if battle.weather else "none", 0) / 4.0,

        # --- Team alive counts (0 to 6) ---
        sum(1 for mon in battle.team.values() if not mon.fainted) / 6.0,
        sum(1 for mon in battle.opponent_team.values() if not mon.fainted) / 6.0,
    ]

    features = features.extend(extract_side_condition_features(battle))

    return np.array(features, dtype=np.float32)


def extract_action_mask(battle: Battle) -> np.ndarray:
    """Action space representation:

    Indices 0-3: Move 1 to 4
    Indices 4-8: Switch to teammate 1 to 5
    """
    mask = np.zeros(9, dtype=np.float32)
    p1_active = battle.active_pokemon

    # 1. Moves available
    if p1_active and p1_active.moves:
        for idx, move in enumerate(list(p1_active.moves.values())[:4]):
            if move.current_pp > 0:
                mask[idx] = 1.0

    # 2. Legal switches
    teammates = [mon for mon in battle.team.values() if mon != p1_active]
    for idx, mon in enumerate(teammates[:5]):
        if not mon.fainted:
            mask[4 + idx] = 1.0

    return mask


def parse_replay_log(raw_log: str, p1_username: str):
    """Replays raw Showdown log protocol and yields (state, action_mask, turn_number)."""
    # Initialize battle state tracker
    battle = Battle(
        battle_tag="replay_battle",
        username=p1_username,
        logger=logger,
        gen=GEN,
    )

    lines = raw_log.split("\n")
    trajectory = []

    for line in lines:
        print(line)
        if not line.strip():
            continue

        parts = line.split("|")
        command = parts[1] if len(parts) > 1 else ""

        # Feed the raw protocol line to poke-env's internal parser
        try:
            battle.parse_message(parts[1:])
        except Exception:
            # Replay messages occasionally include non-standard chat or cosmetic tokens
            continue

        # Whenever a new turn starts, snapshot state and action possibilities
        if command == "turn":
            turn_num = int(parts[2])
            state_vec = extract_state_vector(battle)
            action_mask = extract_action_mask(battle)

            trajectory.append({
                "turn": turn_num,
                "state": state_vec,
                "mask": action_mask
            })

    return trajectory
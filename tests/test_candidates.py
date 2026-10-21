import math
import unittest

from src.models.candidates import (
    FEATURE_NAMES,
    FrequencyTables,
    candidate_actions,
    candidate_features,
)


def _row(**overrides):
    row = {
        "battle_id": "b1",
        "turn_number": 5,
        "active_species": "Garchomp",
        "active_hp": 0.8,
        "active_status": "none",
        "active_last_move": "Earthquake",
        "active_revealed_moves": ["Earthquake"],
        "opponent_species": "Corviknight",
        "opponent_hp": 1.0,
        "opponent_status": "none",
        "opponent_revealed_moves": ["Brave Bird"],
        "team_remaining": 6,
        "opponent_remaining": 6,
        "available_switches": ["Heatran", "Toxapex"],
        "bench_hp": {"Heatran": 0.5, "Toxapex": 1.0},
        "bench_status": {"Heatran": "none", "Toxapex": "tox"},
        "label_action_type": "move",
        "label_target": "Earthquake",
    }
    for stat in ("atk", "def", "spa", "spd", "spe"):
        row[f"active_{stat}_boost"] = 0
        row[f"opponent_{stat}_boost"] = 0
    for hazard in ("stealth_rock", "spikes", "toxic_spikes", "sticky_web"):
        row[f"own_{hazard}"] = 0
        row[f"opponent_{hazard}"] = 0
    row.update(overrides)
    return row


def _tables(*rows):
    tables = FrequencyTables()
    for row in rows:
        tables.add(row)
    return tables


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tables = _tables(
            _row(label_target="Earthquake"),
            _row(label_target="Swords Dance"),
            _row(label_target="Scale Shot"),
            _row(label_action_type="switch", label_target="Heatran"),
        )

    def test_candidates_combine_revealed_and_common_moves_with_bench(self):
        actions = candidate_actions(_row(), self.tables, move_k=10)
        self.assertEqual(
            actions,
            [
                ("move", "Earthquake"),
                ("move", "Swords Dance"),
                ("move", "Scale Shot"),
                ("switch", "Heatran"),
                ("switch", "Toxapex"),
            ],
        )

    def test_full_revealed_moveset_excludes_other_moves(self):
        revealed = ["Earthquake", "Fire Fang", "Stealth Rock", "Dragon Tail"]
        actions = candidate_actions(_row(active_revealed_moves=revealed), self.tables)
        self.assertEqual([target for kind, target in actions if kind == "move"], revealed)

    def test_out_of_fold_tables_remove_only_that_fold(self):
        fold = _tables(_row(label_target="Swords Dance"))
        out_of_fold = self.tables - fold
        self.assertEqual(out_of_fold.move_species["Garchomp"]["Swords Dance"], 0)
        self.assertEqual(out_of_fold.move_species["Garchomp"]["Earthquake"], 1)
        self.assertEqual(out_of_fold.switch_species["Garchomp"]["Heatran"], 1)

    def test_features_cover_matchups_and_leave_other_kind_missing(self):
        actions = [("move", "Earthquake"), ("switch", "Heatran")]
        move_row, switch_row = candidate_features(_row(), actions, self.tables)
        self.assertEqual(len(move_row), len(FEATURE_NAMES))
        feature = dict(zip(FEATURE_NAMES, move_row))
        self.assertEqual(feature["move_is_last"], 1.0)
        self.assertEqual(feature["move_effectiveness"], 0.0)  # Ground vs Flying
        self.assertTrue(math.isnan(feature["switch_hp"]))
        feature = dict(zip(FEATURE_NAMES, switch_row))
        self.assertEqual(feature["is_switch"], 1.0)
        self.assertEqual(feature["switch_hp"], 0.5)
        self.assertEqual(feature["switch_threatened"], 0.5)  # Flying/Steel vs Fire/Steel
        self.assertTrue(math.isnan(feature["move_is_last"]))


if __name__ == "__main__":
    unittest.main()

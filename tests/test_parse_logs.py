import unittest

from src.data.parse_logs import parse_battle
from src.data.prepare_dataset import split_for_battle


REPLAY = """|player|p1|Alice|avatar|1400
|player|p2|Bob|avatar|1450
|teamsize|p1|2
|teamsize|p2|2
|poke|p1|Landorus-Therian, M|
|poke|p1|Heatran, F|
|poke|p2|Ferrothorn, M|
|poke|p2|Zapdos|
|start
|switch|p1a: Landorus|Landorus-Therian, M|100/100
|switch|p2a: Ferrothorn|Ferrothorn, M|100/100
|turn|1
|move|p1a: Landorus|U-turn|p2a: Ferrothorn
|-damage|p2a: Ferrothorn|80/100
|switch|p1a: Heatran|Heatran, F|100/100|[from] U-turn
|move|p2a: Ferrothorn|Spikes|p1a: Heatran
|-sidestart|p1: Alice|Spikes
|upkeep
|turn|2
|switch|p2a: Zapdos|Zapdos|100/100
|move|p1a: Heatran|Magma Storm|p2a: Zapdos
|-damage|p2a: Zapdos|0 fnt
|faint|p2a: Zapdos
|switch|p2a: Ferrothorn|Ferrothorn, M|80/100
|turn|3
|move|p1a: Heatran|Taunt|p2a: Ferrothorn
|move|p2a: Ferrothorn|Power Whip|p1a: Heatran
"""


class ParseBattleTests(unittest.TestCase):
    def setUp(self):
        self.examples = parse_battle(
            {
                "id": "gen8ou-test",
                "formatid": "gen8ou",
                "rating": 1425,
                "log": REPLAY,
            }
        )

    def test_extracts_one_deliberate_action_per_side_and_turn(self):
        labels = [
            (row["turn_number"], row["actor_side"], row["label_action_type"], row["label_target"])
            for row in self.examples
        ]
        self.assertEqual(
            labels,
            [
                (1, "p1", "move", "U-turn"),
                (1, "p2", "move", "Spikes"),
                (2, "p2", "switch", "Zapdos"),
                (2, "p1", "move", "Magma Storm"),
                (3, "p1", "move", "Taunt"),
                (3, "p2", "move", "Power Whip"),
            ],
        )

    def test_uses_state_from_start_of_turn(self):
        p1_turn_three = next(
            row
            for row in self.examples
            if row["turn_number"] == 3 and row["actor_side"] == "p1"
        )
        self.assertEqual(p1_turn_three["active_species"], "Heatran")
        self.assertEqual(p1_turn_three["opponent_species"], "Ferrothorn")
        self.assertEqual(p1_turn_three["opponent_remaining"], 1)
        self.assertEqual(p1_turn_three["own_spikes"], 1)
        self.assertEqual(p1_turn_three["team_species"], ["Landorus-Therian", "Heatran"])
        self.assertEqual(p1_turn_three["available_switches"], ["Landorus-Therian"])

    def test_pivot_and_forced_replacement_are_not_labels(self):
        switch_targets = [
            row["label_target"]
            for row in self.examples
            if row["label_action_type"] == "switch"
        ]
        self.assertEqual(switch_targets, ["Zapdos"])

    def test_switch_target_uses_hidden_team_preview_form(self):
        replay = """|poke|p1|Urshifu-*|
|poke|p1|Heatran|
|poke|p2|Zapdos|
|poke|p2|Ferrothorn|
|start
|switch|p1a: Heatran|Heatran|100/100
|switch|p2a: Zapdos|Zapdos|100/100
|turn|1
|switch|p1a: Urshifu|Urshifu-Rapid-Strike|100/100
|move|p2a: Zapdos|Discharge|p1a: Urshifu
"""
        examples = parse_battle(
            {"id": "hidden-form", "formatid": "gen8ou", "rating": 1400, "log": replay}
        )
        switch = next(row for row in examples if row["label_action_type"] == "switch")
        self.assertEqual(switch["label_target"], "Urshifu-*")


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic(self):
        first = split_for_battle("battle-123", seed=42)
        self.assertEqual(first, split_for_battle("battle-123", seed=42))
        self.assertIn(first, {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()

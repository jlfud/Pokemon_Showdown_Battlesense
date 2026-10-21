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

    def test_hidden_form_active_and_fainted_are_not_available_switches(self):
        replay = """|poke|p1|Urshifu-*|
|poke|p1|Heatran|
|poke|p1|Clefable|
|poke|p2|Zapdos|
|start
|switch|p1a: Urshifu|Urshifu-Rapid-Strike|100/100
|switch|p2a: Zapdos|Zapdos|100/100
|turn|1
|move|p1a: Urshifu|Surging Strikes|p2a: Zapdos
|move|p2a: Zapdos|Hurricane|p1a: Urshifu
|-damage|p1a: Urshifu|0 fnt
|faint|p1a: Urshifu
|switch|p1a: Heatran|Heatran|100/100
|turn|2
|move|p1a: Heatran|Magma Storm|p2a: Zapdos
"""
        examples = parse_battle(
            {"id": "hidden-bench", "formatid": "gen8ou", "rating": 1400, "log": replay}
        )
        bench = {
            row["turn_number"]: row["available_switches"]
            for row in examples
            if row["actor_side"] == "p1"
        }
        self.assertEqual(bench[1], ["Heatran", "Clefable"])
        self.assertEqual(bench[2], ["Clefable"])

    def test_item_and_ability_forced_switches_are_not_labels(self):
        replay = """|poke|p1|Toxapex|
|poke|p1|Tapu Lele|
|poke|p1|Golisopod|
|poke|p2|Urshifu-*|
|start
|switch|p1a: Toxapex|Toxapex|100/100
|switch|p2a: Urshifu|Urshifu|100/100
|turn|1
|move|p2a: Urshifu|Close Combat|p1a: Toxapex
|-damage|p1a: Toxapex|60/100
|-enditem|p1a: Toxapex|Eject Button
|switch|p1a: Golisopod|Golisopod|100/100
|turn|2
|move|p2a: Urshifu|Wicked Blow|p1a: Golisopod
|-damage|p1a: Golisopod|40/100
|-activate|p1a: Golisopod|ability: Emergency Exit
|switch|p1a: Tapu Lele|Tapu Lele|100/100
|turn|3
|move|p2a: Urshifu|U-turn|p1a: Tapu Lele
|-enditem|p1a: Tapu Lele|Eject Button|[from] move: Knock Off
|switch|p1a: Golisopod|Golisopod|40/100
"""
        examples = parse_battle(
            {"id": "forced", "formatid": "gen8ou", "rating": 1400, "log": replay}
        )
        p1_labels = [
            (row["turn_number"], row["label_action_type"], row["label_target"])
            for row in examples
            if row["actor_side"] == "p1"
        ]
        self.assertEqual(p1_labels, [(3, "switch", "Golisopod")])

    def test_unselected_moves_are_not_labels_but_reveal_moveset(self):
        replay = """|poke|p1|Dragapult|
|poke|p1|Heatran|
|poke|p2|Hatterene|
|start
|switch|p1a: Dragapult|Dragapult|100/100
|switch|p2a: Hatterene|Hatterene|100/100
|turn|1
|move|p1a: Dragapult|Phantom Force||[still]
|-prepare|p1a: Dragapult|Phantom Force
|move|p2a: Hatterene|Calm Mind|p2a: Hatterene
|turn|2
|move|p1a: Dragapult|Phantom Force|p2a: Hatterene|[from] lockedmove
|move|p2a: Hatterene|Psychic|p1a: Dragapult
|turn|3
|move|p1a: Dragapult|Dragon Dance|p1a: Dragapult
|move|p1a: Dragapult|Dragon Dance|p1a: Dragapult|[from]ability: Dancer
|move|p2a: Hatterene|Mystical Fire|p1a: Dragapult
|turn|4
|switch|p1a: Heatran|Heatran|100/100
|move|p2a: Hatterene|Psychic|p1a: Heatran
|turn|5
|switch|p1a: Dragapult|Dragapult|100/100
"""
        examples = parse_battle(
            {"id": "locked", "formatid": "gen8ou", "rating": 1400, "log": replay}
        )
        p1 = {row["turn_number"]: row for row in examples if row["actor_side"] == "p1"}
        self.assertEqual(sorted(p1), [1, 3, 4, 5])
        self.assertEqual(p1[3]["active_last_move"], "Phantom Force")
        self.assertEqual(p1[3]["active_revealed_moves"], ["Phantom Force"])
        self.assertEqual(p1[4]["active_last_move"], "Dragon Dance")
        self.assertEqual(p1[4]["active_revealed_move_count"], 2)

    def test_revealed_moves_persist_across_switches_but_last_move_resets(self):
        ferrothorn_return = next(
            row
            for row in parse_battle(
                {"id": "gen8ou-test", "formatid": "gen8ou", "rating": 1425, "log": REPLAY}
            )
            if row["turn_number"] == 3 and row["actor_side"] == "p2"
        )
        self.assertEqual(ferrothorn_return["active_species"], "Ferrothorn")
        self.assertEqual(ferrothorn_return["active_last_move"], "none")
        self.assertEqual(ferrothorn_return["active_revealed_moves"], ["Spikes"])


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic(self):
        first = split_for_battle("battle-123", seed=42)
        self.assertEqual(first, split_for_battle("battle-123", seed=42))
        self.assertIn(first, {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()

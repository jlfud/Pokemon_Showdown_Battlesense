import unittest

from src.models.model_manager import SpecificActionBaseline


ARTIFACT = {
    "global_counts": {
        "move": {"Earthquake": 30, "Swords Dance": 10},
        "switch": {"Corviknight": 20, "Clefable": 10},
    },
    "species_counts": {
        "move": {"Garchomp": {"Earthquake": 20, "Swords Dance": 10}},
        "switch": {"Garchomp": {"Corviknight": 12, "Clefable": 4}},
    },
    "matchup_counts": {
        "move": {"Garchomp\tHeatran": {"Earthquake": 18, "Swords Dance": 2}},
        "switch": {"Garchomp\tWeavile": {"Corviknight": 15, "Clefable": 5}},
    },
}


class SpecificActionBaselineTests(unittest.TestCase):
    def setUp(self):
        self.model = SpecificActionBaseline(ARTIFACT)

    def test_move_distribution_uses_matchup_and_normalizes(self):
        distribution = self.model.distribution("move", "Garchomp", "Heatran")
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreater(distribution["Earthquake"], distribution["Swords Dance"])

    def test_switch_distribution_masks_illegal_targets(self):
        distribution = self.model.distribution(
            "switch",
            "Garchomp",
            "Weavile",
            candidates=["Clefable", "Toxapex"],
        )
        self.assertEqual(set(distribution), {"Clefable", "Toxapex"})
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_move_distribution_favors_last_and_revealed_moves(self):
        model = SpecificActionBaseline(
            {
                **ARTIFACT,
                "reveal_counts": {
                    "2:last": {"last": 60, "revealed": 25, "unrevealed": 15},
                },
            }
        )
        distribution = model.distribution(
            "move",
            "Garchomp",
            "Heatran",
            revealed_moves=["Swords Dance", "Scale Shot"],
            last_move="Swords Dance",
        )
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertIn("Scale Shot", distribution)
        self.assertAlmostEqual(distribution["Swords Dance"], 61 / 103)
        self.assertAlmostEqual(distribution["Scale Shot"], 26 / 103)
        self.assertAlmostEqual(distribution["Earthquake"], 16 / 103)


if __name__ == "__main__":
    unittest.main()

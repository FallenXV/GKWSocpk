from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from socpk_gui import classify_columns, collection_summary, load_collections


class DashboardDataTests(unittest.TestCase):
    def test_classifies_all_supported_schemas(self) -> None:
        self.assertEqual(
            classify_columns(["CPU", "Board_Power_W", "GB6_Multi_Score"]),
            "CPU",
        )
        self.assertEqual(
            classify_columns(["GPU", "Board_Power_W", "GPU_Score"]),
            "GPU",
        )
        self.assertEqual(
            classify_columns(["brand", "model", "minutes", "capacityWh", "avgPowerW"]),
            "Battery",
        )
        self.assertIsNone(classify_columns(["unrelated", "columns"]))

    def test_loads_merges_and_deduplicates_collections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cpu = pd.DataFrame(
                {
                    "CPU": ["Chip A", "Chip A"],
                    "Board_Power_W": [2.0, 4.0],
                    "GB6_Multi_Score": [2000, 3600],
                }
            )
            cpu.to_csv(root / "cpu.csv", index=False)
            (root / "archive").mkdir()
            cpu.to_csv(root / "archive" / "cpu-copy.csv", index=False)
            pd.DataFrame(
                {
                    "brand": ["Example"],
                    "model": ["Phone"],
                    "os": ["OS 1"],
                    "minutes": [600],
                    "capacityWh": [20],
                    "avgPowerW": [2.0],
                }
            ).to_csv(root / "battery.csv", index=False)

            collections, warnings = load_collections(root)

            self.assertEqual(warnings, [])
            self.assertEqual(set(collections), {"CPU", "Battery"})
            self.assertEqual(len(collections["CPU"]), 2)
            self.assertAlmostEqual(collections["CPU"]["Efficiency"].iloc[0], 1000.0)
            self.assertEqual(
                collection_summary(collections),
                {"CPU": (1, 2), "Battery": (1, 1)},
            )


if __name__ == "__main__":
    unittest.main()

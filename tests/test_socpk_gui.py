from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from socpk_gui import (
    add_geekerwan_capacity_overlay,
    classify_columns,
    collection_summary,
    load_collections,
)


class DashboardDataTests(unittest.TestCase):
    def test_adds_geekerwan_capacity_overlay_without_replacing_source_data(self) -> None:
        source = pd.DataFrame(
            {
                "brand": ["一加", "Samsung", "Example"],
                "model": ["Ace6 至尊版", "S26 Ultra", "Phone"],
                "capacityWh": [31.6, 18.94, 20.0],
                "avgPowerW": [3.16, 1.894, 2.0],
                "minPerWh": [20.0, 30.0, 25.0],
            }
        )

        overlaid = add_geekerwan_capacity_overlay(source)

        self.assertEqual(overlaid.loc[0, "geekerwanAdvertisedMah"], 8600)
        self.assertEqual(overlaid.loc[0, "geekerwanMeasuredMah"], 8202)
        self.assertAlmostEqual(overlaid.loc[0, "geekerwanShortfallPct"], 4.62790698)
        self.assertAlmostEqual(overlaid.loc[0, "geekerwanCapacityWh"], 31.6 * 8202 / 8600)
        self.assertAlmostEqual(overlaid.loc[0, "geekerwanAvgPowerW"], 3.16 * 8202 / 8600)
        self.assertAlmostEqual(overlaid.loc[0, "geekerwanMinPerWh"], 20.0 * 8600 / 8202)
        self.assertEqual(overlaid.loc[0, "capacityWh"], source.loc[0, "capacityWh"])
        self.assertEqual(overlaid.loc[1, "geekerwanShortfallMah"], 225)
        self.assertTrue(pd.isna(overlaid.loc[2, "geekerwanMeasuredMah"]))

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

    def test_separates_mobile_and_laptop_gpu_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pd.DataFrame(
                {
                    "GPU": ["Mobile A", "Mobile A"],
                    "Board_Power_W": [2.0, 4.0],
                    "GPU_Score": [600, 1000],
                    "Efficiency": [300, 250],
                }
            ).to_csv(root / "gpu_curves.csv", index=False)
            pd.DataFrame(
                {
                    "GPU": ["RTX Example", "RTX Example"],
                    "Board_Power_W": [80.0, 140.0],
                    "GPU_Score": [12000, 18000],
                    "Efficiency": [150, 128.57],
                }
            ).to_csv(root / "laptop_gpu_curves.csv", index=False)

            collections, warnings = load_collections(root)

            self.assertEqual(warnings, [])
            self.assertEqual(set(collections), {"GPU", "Laptop GPU"})
            self.assertEqual(collections["GPU"]["__label"].unique().tolist(), ["Mobile A"])
            self.assertEqual(
                collections["Laptop GPU"]["__label"].unique().tolist(),
                ["RTX Example"],
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from cpu_benchmarks import CPU_BENCHMARKS

from socpk_gui import (
    add_geekerwan_capacity_overlay,
    classify_columns,
    collection_summary,
    load_collections,
    ComparisonDashboard,
    DATASET_DEFINITIONS,
    CoreFilter,
)


class DashboardDataTests(unittest.TestCase):
    def test_leader_keeps_full_name_and_core_type(self):
        label = "Snapdragon 8 Elite Gen 5 — Oryon V3 M · medium · e"
        frame = pd.DataFrame({"__label": [label], "CPU": ["Snapdragon 8 Elite Gen 5"],
                              "Core": ["Oryon V3 M"], "Core_Group": ["medium"], "Core_Variant": ["e"]})
        self.assertEqual(ComparisonDashboard._leader_label(label, frame),
                         "Snapdragon 8 Elite Gen 5\nOryon V3 M · Medium · E-core")
        self.assertEqual(ComparisonDashboard._leader_label(label, frame.iloc[:0]), label)

    def test_core_filter_scopes_chart_and_top_five_without_losing_other_selections(self):
        dashboard = ComparisonDashboard.__new__(ComparisonDashboard)
        dashboard.dataset_key = "SPEC INT"
        frame = pd.DataFrame({"__label": ["Chip P", "Chip E", "Other E"],
                              "Core": ["Everest", "Sawtooth", "Cortex-A720"],
                              "Core_Group": ["super", "medium", "medium"],
                              "Efficiency": [100., 10., 20.]})
        dashboard.collections = {"SPEC INT": frame, "CPU GB7": frame.drop(columns=["Core", "Core_Group"])}
        dashboard.selected = {"SPEC INT": set(frame["__label"])}
        dashboard.core_filters = {"SPEC INT": CoreFilter(frozenset({"medium"}))}
        dashboard.selection_modes = {"SPEC INT": "manual"}
        dashboard.selection_buttons = {}
        self.assertEqual(set(dashboard._selected_frame()["__label"]), {"Chip E", "Other E"})
        self.assertEqual(dashboard._ranked_labels("SPEC INT"), ["Other E", "Chip E"])
        self.assertIn("Chip P", dashboard.selected["SPEC INT"])
        dashboard.visible_labels = ["Chip E"]  # Search narrows the group further.
        dashboard.refresh_profile_list, dashboard.draw_charts = dashboard._apply_selection_mode, Mock()
        dashboard.select_top_five()
        self.assertEqual(dashboard.selected["SPEC INT"], {"Chip E"})
        dashboard.core_filters["SPEC INT"] = CoreFilter(frozenset({"medium"}), frozenset({"Cortex-A720"}))
        self.assertEqual(dashboard._core_filtered_frame("SPEC INT")["__label"].tolist(), ["Other E"])
        self.assertEqual(len(dashboard._core_filtered_frame("CPU GB7")), 3)

    def test_multiselect_groups_names_and_latched_selection_modes(self):
        dashboard = ComparisonDashboard.__new__(ComparisonDashboard)
        dashboard.dataset_key = "SPEC INT"
        frame = pd.DataFrame({"__label": [f"Chip {i}" for i in range(8)],
                              "Core_Group": ["super"] * 3 + ["medium"] * 3 + ["small"] * 2,
                              "Core": ["Core A", "Core B"] * 4,
                              "Efficiency": list(range(8)), "SPEC2026_INT_Score": list(range(8, 0, -1))})
        dashboard.collections = {"SPEC INT": frame}
        dashboard.selected = {"SPEC INT": set()}
        dashboard.selection_modes = {"SPEC INT": "top5"}
        dashboard.selection_buttons = {}
        dashboard.core_filters = {}
        dashboard._sync_core_filters, dashboard.draw_charts = Mock(), Mock()
        def refresh():
            dashboard.visible_labels = dashboard._core_filtered_frame("SPEC INT")["__label"].tolist()
            dashboard._apply_selection_mode()
        dashboard.refresh_profile_list = refresh
        dashboard.set_core_group("super")
        self.assertEqual(dashboard.selected["SPEC INT"], {"Chip 0", "Chip 1", "Chip 2"})
        dashboard.set_core_group("medium")
        self.assertEqual(dashboard.core_filters["SPEC INT"].groups, {"super", "medium"})
        self.assertEqual(dashboard.selected["SPEC INT"], {f"Chip {i}" for i in range(1, 6)})
        dashboard.select_visible()
        self.assertEqual(len(dashboard.selected["SPEC INT"]), 6)
        dashboard.toggle_core_name("Core A")
        self.assertEqual(dashboard.selected["SPEC INT"], {"Chip 0", "Chip 2", "Chip 4"})
        dashboard.toggle_core_name("Core B")
        self.assertEqual(len(dashboard.selected["SPEC INT"]), 6)
        dashboard.set_core_group("super")  # Toggle one group off; All shown stays latched.
        self.assertEqual(dashboard.selected["SPEC INT"], {"Chip 3", "Chip 4", "Chip 5"})
        dashboard.clear_selection()
        dashboard.set_core_group("")
        self.assertEqual(dashboard.selected["SPEC INT"], set())
        dashboard.select_top_five()
        dashboard.view_var = Mock()
        dashboard.view_var.get.return_value = "Performance curve"
        dashboard.on_view_change()
        self.assertEqual(dashboard.selected["SPEC INT"], {f"Chip {i}" for i in range(5)})

    def test_new_cpu_schemas_keep_benchmarks_and_core_groups_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, benchmark in CPU_BENCHMARKS.items():
                frame = pd.DataFrame({"CPU": ["Chip", "Chip"], "Board_Power_W": [1., 1.],
                                      benchmark.score_column: [1.23456, 2.34567], "Benchmark": name})
                if benchmark.single_core:
                    frame["Core"] = "Unknown"
                    frame["Core_Group"] = ["large", "small"]
                frame.to_csv(root / benchmark.filename, index=False)
            collections, warnings = load_collections(root)
            self.assertEqual(warnings, [])
            self.assertEqual(set(collections), {b.dataset_key for b in CPU_BENCHMARKS.values()})
            for kind in ("SPEC INT", "SPEC FP"):
                frame = collections[kind]
                self.assertEqual(frame["__label"].tolist(), ["Chip — Unknown · large", "Chip — Unknown · small"])
                self.assertAlmostEqual(frame["Efficiency"].iloc[0], 1.23456)

    def test_rejects_mixed_cpu_scores_and_mislabeled_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"CPU": ["Chip"], "Board_Power_W": [1], "GB6_Multi_Score": [6000],
                          "GB7_Multi_Score": [7000]}).to_csv(root / "mixed.csv", index=False)
            pd.DataFrame({"CPU": ["Chip"], "Board_Power_W": [1], "GB6_Multi_Score": [6000],
                          "Benchmark": ["GB7"]}).to_csv(root / "wrong.csv", index=False)
            collections, warnings = load_collections(root)
            self.assertEqual(collections, {})
            self.assertEqual(len(warnings), 2)

    def test_new_benchmark_chart_views_rank_and_export_with_correct_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("GB7", "SPEC2026_INT", "SPEC2026_FP"):
                benchmark = CPU_BENCHMARKS[name]
                frame = pd.DataFrame({"CPU": ["Chip"] * 4, "Board_Power_W": [.5, 1., .5, 1.],
                                      benchmark.score_column: [1.23456, 2.34567, 2., 3.]})
                if benchmark.single_core:
                    frame["Core"] = "Unknown"
                    frame["Core_Group"] = ["large", "large", "small", "small"]
                target = root / benchmark.filename
                frame.to_csv(target, index=False)
                collections, warnings = load_collections(root, [target])
                self.assertEqual(warnings, [])
                kind = benchmark.dataset_key
                frame = collections[kind]
                dashboard = ComparisonDashboard.__new__(ComparisonDashboard)
                dashboard.dataset_key = kind
                dashboard.collections = collections
                dashboard.selected = {kind: set(frame["__label"])}
                dashboard.figure = Figure(figsize=(12, 6))
                FigureCanvasAgg(dashboard.figure)
                dashboard.main_axis, dashboard.rank_axis = dashboard.figure.subplots(1, 2)
                dashboard.ranking_page_size = 9
                dashboard.chart_note = Mock()
                dashboard.source_note = Mock()
                dashboard.stat_values = {key: Mock() for key in ("profiles", "points", "leader")}
                dashboard.load_warnings = []
                dashboard.view_var = Mock()
                for view in DATASET_DEFINITIONS[kind].views:
                    dashboard.view_var.get.return_value = view
                    dashboard.hover_points, dashboard.line_artists = [], []
                    dashboard._draw_curve_charts(frame)
                    dashboard._update_stats(frame)
                    metric = benchmark.score_column if view == "Performance curve" else "Efficiency"
                    expected = frame.groupby("__label")[metric].max().sort_values(ascending=False)
                    pd.testing.assert_series_equal(dashboard.ranking_values, expected)
                    if benchmark.single_core:
                        self.assertIn("1.235", dashboard.hover_points[0].details)
                        self.assertIn("Unknown", dashboard._ranking_label(next(iter(dashboard.selected[kind]))))
                        lines = [line for line in dashboard.main_axis.lines if hasattr(line, "_socpk_label")]
                        self.assertEqual(len(lines), 2)
                        for line in lines:
                            self.assertEqual(line.get_linestyle(), ":")
                            self.assertEqual(len(line.get_xdata()), 2)
                        for text in dashboard.main_axis.get_legend().get_texts():
                            self.assertNotIn("large", text.get_text())
                            self.assertNotIn("small", text.get_text())
                    dashboard.figure.savefig(root / f"{name}-{view}.png")

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

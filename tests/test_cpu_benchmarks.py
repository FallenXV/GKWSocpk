from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cpu_benchmarks import CPU_BENCHMARKS

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "Performance Benchmark" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu_parser = load_module("cpu_curve_parser")


def chart(slug):
    spec = slug == "mobile-soc-spec26"
    series = []
    for suite in (["INT", "FP"] if spec else ["GB6", "GB7"]):
        for group in (["large", "small"] if spec else [""]):
            score = {"INT": 1.23456, "FP": 2.34567, "GB6": 6000, "GB7": 7000}[suite]
            series.append({
                "name": "芯片", "name_en": "Chip", "id": len(series),
                "meta": {"suite": suite, "coreName": "Unknown", "coreGroup": group},
                "points": [[0.5, score], [1.0, score * 1.5]],
            })
    return {"type": "line-chart", "config": {"xUnit": "W"}, "series": series}


class CpuBenchmarkTests(unittest.TestCase):
    def test_each_benchmark_keeps_its_scores_and_all_spec_cores(self):
        with patch.object(cpu_parser, "fetch_chart_page", side_effect=chart):
            for name, definition in CPU_BENCHMARKS.items():
                with self.subTest(benchmark=name):
                    frame = cpu_parser.scrape_cpu_curves(["Chip"], benchmark=name)
                    expected_rows = 4 if definition.single_core else 2
                    self.assertEqual(len(frame), expected_rows)
                    self.assertEqual(set(frame["Benchmark"]), {name})
                    self.assertEqual(set(frame.columns) & {b.score_column for b in CPU_BENCHMARKS.values()},
                                     {definition.score_column})
                    self.assertAlmostEqual(frame["Efficiency"].iloc[0], frame[definition.score_column].iloc[0] / 0.5)
                    if definition.single_core:
                        self.assertEqual(set(frame["Core_Group"]), {"large", "small"})
                        self.assertEqual(set(frame["Core"]), {"Unknown"})
                        self.assertEqual(set(frame["Core_Variant"]), {""})

    def test_all_benchmarks_share_two_consistent_page_polls(self):
        with patch.object(cpu_parser, "fetch_chart_page", side_effect=chart) as fetch:
            frames = cpu_parser.scrape_cpu_benchmarks()
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(set(frames), set(CPU_BENCHMARKS))
        self.assertEqual(frames["GB6"]["GB6_Multi_Score"].iloc[0], 6000)
        self.assertEqual(frames["GB7"]["GB7_Multi_Score"].iloc[0], 7000)

    def test_core_filter_and_unique_chip_discovery(self):
        with patch.object(cpu_parser, "fetch_chart_page", side_effect=chart):
            frame = cpu_parser.scrape_cpu_curves(benchmark="SPEC2026_FP", core_group="small")
            self.assertEqual(set(frame["Core_Group"]), {"small"})
            self.assertEqual(cpu_parser.discover_cpu_names("SPEC2026_INT"), ["Chip"])
            with self.assertRaises(ValueError):
                cpu_parser.scrape_cpu_curves(benchmark="GB7", core_group="small")
            with self.assertRaises(ValueError):
                cpu_parser.parse_cpu_curve("Chip", "https://example.com/svg", benchmark="GB7")

    def test_missing_spec_core_identity_fails(self):
        page = chart("mobile-soc-spec26")
        page["series"][0]["meta"].pop("coreGroup")
        with patch.object(cpu_parser, "fetch_chart_page", return_value=page), self.assertRaises(ValueError):
            cpu_parser.scrape_cpu_curves(benchmark="SPEC2026_INT")

    def test_all_cli_exports_distinct_files_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / CPU_BENCHMARKS["GB6"].filename
            existing.write_bytes(b"historical\r\n")
            with patch.object(cpu_parser, "fetch_chart_page", side_effect=chart), \
                 patch("sys.argv", ["cpu_curve_parser.py", "--benchmark", "all", "--output-dir", directory]):
                cpu_parser.main()
            self.assertEqual(existing.read_bytes(), b"historical\r\n")
            files = [p for p in Path(directory).glob("*.csv") if p != existing]
            self.assertEqual(len(files), 4)
            self.assertEqual({pd.read_csv(p)["Benchmark"].iloc[0] for p in files}, set(CPU_BENCHMARKS))

    def test_standalone_analysis_preserves_spec_core_profiles(self):
        analysis = load_module("curve_analysis")
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cpu_parser, "fetch_chart_page", side_effect=chart):
            for name in ("GB7", "SPEC2026_INT", "SPEC2026_FP"):
                path = Path(directory) / f"{name}.csv"
                cpu_parser.scrape_cpu_curves(benchmark=name).to_csv(path, index=False)
                frame, kind = analysis.load_and_normalize(str(path))
                self.assertEqual(kind, "CPU")
                expected_profiles = 1 if name == "GB7" else 2
                self.assertEqual(frame["Model"].nunique(), expected_profiles)
                self.assertEqual(len(analysis.compute_statistics(frame)), expected_profiles)


if __name__ == "__main__":
    unittest.main()

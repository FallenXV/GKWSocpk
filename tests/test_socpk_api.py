from __future__ import annotations

import base64
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from Battery import battery_parser
from socpk_client import (
    battery_rows_from_page,
    decode_chart_blob,
    fetch_chart_page,
    fetch_curve_series,
    new_snapshot,
)


# Public API examples captured 2026-09-09. Expected values were independently
# checked against the WebAssembly codec shipped by the site, not an encoder
# that mirrors the Python implementation.
LINE_BLOB = "AUT4YJQD4+mVEoSVoDPwjNkEsPSrFveZvwqskds2"
BAR_BLOB = "Ao7jr8UB3cTHGw=="
LINE_POINTS = [
    [0.510011716192924, 1591.9999587068894],
    [2.2399840069998556, 4762.00004531778],
    [13.239993574088512, 11110.999961461117],
]


def response(payload, status=200):
    result = Mock(status_code=status)
    result.json.return_value = payload
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(str(status))
    return result


def page():
    return {
        "type": "line-chart", "config": {"xUnit": "W"},
        "dataToken": "public-token", "dataExp": 2000, "dataVersion": "one",
        "series": [{"id": 7, "name": "A19 Pro", "meta": {"suite": "GB6"}}],
    }


def chart_data():
    return {"v": 1, "dataVersion": "one", "series": [{"id": 7, "p": LINE_BLOB}]}


class ChartCodecTests(unittest.TestCase):
    def test_line_and_scalar_match_site_codec(self):
        self.assertEqual(decode_chart_blob(LINE_BLOB), LINE_POINTS)
        self.assertEqual(decode_chart_blob(BAR_BLOB), 725.9999649188321)

    def test_rejects_bad_version_truncation_varint_and_trailing_data(self):
        raw = base64.b64decode(LINE_BLOB)
        samples = [b"", b"\x03" + raw[1:], raw[:-1], raw + b"\0",
                   raw[:5] + b"\x80" * 9, raw[:5] + b"\x81\x80\x04"]
        for sample in samples:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                decode_chart_blob(base64.b64encode(sample).decode())
        with self.assertRaises(ValueError):
            decode_chart_blob("not base64!")


class ChartApiTests(unittest.TestCase):
    def setUp(self):
        self.session_patch = patch("socpk_client.requests.Session")
        self.session = self.session_patch.start().return_value.__enter__.return_value
        self.addCleanup(self.session_patch.stop)
        self.clock_patch = patch("socpk_client.time.time", return_value=1000)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)

    def test_fetch_joins_by_id_and_sends_public_token(self):
        metadata = page()
        metadata["series"].append({"id": 8, "name": "Other"})
        data = chart_data()
        data["series"].insert(0, {"id": 8, "p": BAR_BLOB})
        self.session.get.side_effect = [response(metadata), response(data)]
        result = fetch_chart_page("test")
        self.assertEqual(result["series"][0]["points"], LINE_POINTS)
        self.assertEqual(result["series"][1]["points"], 725.9999649188321)
        call = self.session.get.call_args_list[1]
        self.assertEqual(call.args[0], "https://www.socpk.com/api/pages/test/data")
        self.assertEqual(call.kwargs["headers"], {"X-Chart-Token": "2000.public-token"})

    def test_refreshes_expiring_token_without_fetching_stale_data(self):
        expired = {**page(), "dataExp": 1005}
        self.session.get.side_effect = [response(expired), response(page()), response(chart_data())]
        self.assertEqual(fetch_chart_page("test")["series"][0]["points"], LINE_POINTS)
        self.assertEqual(self.session.get.call_count, 3)
        self.assertIn("_", self.session.get.call_args_list[1].kwargs["params"])

    def test_refreshes_on_forbidden_or_data_version_mismatch(self):
        for first_data in (response({}, 403), response({**chart_data(), "dataVersion": "old"})):
            with self.subTest(first_data=first_data):
                self.session.get.reset_mock()
                self.session.get.side_effect = [response(page()), first_data,
                                                response(page()), response(chart_data())]
                self.assertEqual(fetch_chart_page("test")["series"][0]["points"], LINE_POINTS)
                self.assertEqual(self.session.get.call_count, 4)

    def test_repeated_mismatch_and_missing_data_fail(self):
        self.session.get.side_effect = [response(page()), response({**chart_data(), "dataVersion": "old"})] * 2
        with self.assertRaisesRegex(ValueError, "changed during fetch"):
            fetch_chart_page("test")
        self.session.get.side_effect = [response(page()), response({**chart_data(), "series": []})]
        with self.assertRaisesRegex(ValueError, "Missing SoCPK data"):
            fetch_chart_page("test")

    def test_separate_polls_do_not_reuse_cached_page(self):
        self.session.get.side_effect = [response(page()), response(chart_data())] * 2
        fetch_chart_page("test")
        fetch_chart_page("test")
        self.assertEqual(self.session.get.call_count, 4)

    def test_cpu_suite_and_legacy_name_selection(self):
        metadata = page()
        metadata["series"] = [
            {"name": "骁龙 8 Gen3", "name_en": "Snapdragon 8 Gen 3",
             "meta": {"suite": suite}, "points": [[2, score]]}
            for suite, score in (("GB6", 5000), ("GB7", 7000))
        ]
        with patch("socpk_client.fetch_chart_page", return_value=metadata):
            result = fetch_curve_series("test", ["SD8 Gen3"], suite="GB6")
            self.assertEqual(result[0]["points"], [[2, 5000]])
            with self.assertRaisesRegex(ValueError, "Unknown or ambiguous"):
                fetch_curve_series("test", ["SD8 Gen3"])
            with self.assertRaisesRegex(ValueError, "Unknown or ambiguous"):
                fetch_curve_series("test", ["unknown"], suite="GB6")

    def test_rejects_invalid_curve_units_and_coordinates(self):
        for points in (1, [[0, 10]], [[1, float("nan")]], [[1]], [["1", 2]]):
            metadata = page()
            metadata["series"][0]["points"] = points
            with patch("socpk_client.fetch_chart_page", return_value=metadata), self.assertRaises(ValueError):
                fetch_curve_series("test")
        metadata["config"]["xUnit"] = "min"
        with patch("socpk_client.fetch_chart_page", return_value=metadata), self.assertRaises(ValueError):
            fetch_curve_series("test")


class BatteryApiTests(unittest.TestCase):
    def test_converts_runtime_energy_and_device_metadata(self):
        metadata = {
            "type": "bar-chart", "config": {"unit": "min"},
            "series": [{"group": "荣耀", "name": "荣耀 WIN RT",
                        "points": decode_chart_blob(BAR_BLOB),
                        "meta": {"systemVersion": "MagicOS 10", "ratedEnergyWh": 36.88,
                                 "videoUrl": "https://example.com/video"}}],
        }
        rows = battery_rows_from_page(metadata)
        self.assertEqual(rows, [["荣耀", "WIN RT", 726, "MagicOS 10", "", 36.88,
                                 "https://example.com/video"]])
        record = battery_parser.rows_to_records(rows, "en")[0]
        self.assertEqual(record["brand"], "Honor")
        self.assertAlmostEqual(record["avgPowerW"], 36.88 * 60 / 726)
        with patch.object(battery_parser, "fetch_chart_page", return_value=metadata) as fetch:
            self.assertEqual(battery_parser.fetch_battery_rows(["https://www.socpk.com/batlife/"]), rows)
            fetch.assert_called_once()
        metadata["series"][0]["meta"].pop("ratedEnergyWh")
        with self.assertRaises(ValueError):
            battery_rows_from_page(metadata)


class SnapshotTests(unittest.TestCase):
    def test_existing_csv_and_json_are_untouched_even_on_repeated_export(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".csv", ".json"):
                original = Path(directory) / ("results" + suffix)
                original.write_bytes(b"historical data\r\n")
                created = []
                for _ in range(2):
                    with new_snapshot(original) as output:
                        output.write("new data")
                        created.append(Path(output.name))
                self.assertEqual(original.read_bytes(), b"historical data\r\n")
                self.assertNotEqual(created[0], created[1])
                self.assertTrue(all(path.read_text() == "new data" for path in created))

    def test_failed_write_removes_only_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "results.csv"
            original.write_text("historical")
            with self.assertRaises(RuntimeError):
                with new_snapshot(original) as output:
                    output.write("incomplete")
                    raise RuntimeError("disk write failed")
            self.assertEqual(list(Path(directory).iterdir()), [original])
            self.assertEqual(original.read_text(), "historical")


class ParserIntegrationTests(unittest.TestCase):
    def test_curve_clis_preserve_requested_existing_output(self):
        root = Path(__file__).resolve().parents[1]
        for filename, kind in (("cpu_curve_parser.py", "CPU"), ("gpu_curve_parser.py", "GPU"),
                               ("laptop_gpu_curve_parser.py", "GPU")):
            spec = importlib.util.spec_from_file_location(filename[:-3], root / "Performance Benchmark" / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            metadata = page()
            metadata["series"][0]["points"] = copy.deepcopy(LINE_POINTS)
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "existing.csv"
                target.write_bytes(b"historical\r\n")
                with patch("socpk_client.fetch_chart_page", return_value=metadata), \
                     patch("sys.argv", [filename, "--output", str(target)]):
                    module.main()
                self.assertEqual(target.read_bytes(), b"historical\r\n")
                snapshots = [path for path in Path(directory).glob("*.csv") if path != target]
                self.assertEqual(len(snapshots), 1)
                self.assertTrue(snapshots[0].read_text().startswith(kind + ","))


if __name__ == "__main__":
    unittest.main()

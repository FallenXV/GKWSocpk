from __future__ import annotations

import json
import shutil
import ssl
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from cpu_benchmarks import CPU_BENCHMARKS
from socpk_data import DATASET_DEFINITIONS, load_collections
from socpk_web import (
    WEB_ROOT,
    DashboardHandler,
    DashboardState,
    build_payload,
    build_servers,
    display_host,
    ensure_certificate,
    leader_label,
    legend_label,
    wrap_with_tls,
)


CPU_CSV = """CPU,Board_Power_W,GB6_Multi_Score,Benchmark
A19 Pro,1.0,1000,GB6
A19 Pro,2.0,1800,GB6
A19 Pro,4.0,3000,GB6
A19 Pro,8.0,4400,GB6
Dimensity 9500,1.5,1200,GB6
Dimensity 9500,6.0,3600,GB6
"""

SPEC_CSV = """CPU,Board_Power_W,SPEC2026_INT_Score,Benchmark,Core,Core_Group,Core_Variant
Snapdragon 8 Elite Gen 5,1.0,2.4,SPEC2026_INT,Oryon V3 L,super,p
Snapdragon 8 Elite Gen 5,2.0,3.6,SPEC2026_INT,Oryon V3 L,super,p
Snapdragon 8 Elite Gen 5,0.5,1.1,SPEC2026_INT,Oryon V3 M,medium,e
Snapdragon 8 Elite Gen 5,1.0,1.8,SPEC2026_INT,Oryon V3 M,medium,e
"""

# 苹果 exercises the brand aliasing that matches Geekerwan's English table;
# X70 is deliberately absent from it and carries no SoC.
BATTERY_CSV = """brand,model,os,minutes,capacityWh,avgPowerW,chipset,soc
苹果,iPhone 17 Pro Max,iOS 27,600,17.5,1.75,Apple A19 Pro,Apple A19 Pro
Xiaomi,17,HyperOS 3,540,27.0,3.0,Qualcomm SM8850-AC Snapdragon 8 Elite Gen 5 (3 nm),Snapdragon 8 Elite Gen 5
Honor,X70,MagicOS 10,700,28.0,2.4,,
"""


def write_snapshots(root: Path, **files: str) -> None:
    (root / "snapshots").mkdir(exist_ok=True)
    for name, body in files.items():
        (root / "snapshots" / f"{name}.csv").write_text(body, encoding="utf-8")


def payload_for(root: Path) -> dict:
    collections, warnings = load_collections(root)
    return build_payload(collections, warnings, root)


def dataset_of(payload: dict, key: str) -> dict:
    return next(entry for entry in payload["datasets"] if entry["key"] == key)


class PayloadTests(unittest.TestCase):
    def test_payload_is_strict_json_with_every_tab_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_snapshots(root, cpu_gb6_curves=CPU_CSV, battery_results=BATTERY_CSV)
            payload = payload_for(root)

            # NaN is not valid JSON, and the browser's parser rejects it.
            json.dumps(payload, allow_nan=False)
            self.assertEqual(
                [entry["key"] for entry in payload["datasets"]],
                list(DATASET_DEFINITIONS),
            )
            self.assertEqual(payload["initialDataset"], "CPU")

    def test_curve_profiles_keep_power_score_and_efficiency_per_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_snapshots(root, cpu_gb6_curves=CPU_CSV)
            entry = dataset_of(payload_for(root), "CPU")

            self.assertEqual(entry["kind"], "curve")
            self.assertEqual(entry["pointCount"], 6)
            self.assertEqual([profile["label"] for profile in entry["profiles"]],
                             ["A19 Pro", "Dimensity 9500"])
            points = entry["profiles"][0]["series"][0]["points"]
            self.assertEqual([point[0] for point in points], [1.0, 2.0, 4.0, 8.0])
            self.assertAlmostEqual(points[0][2], 1000.0)   # efficiency is derived
            self.assertFalse(entry["sparseCurves"])

    def test_spec_profiles_carry_core_metadata_and_shortened_legends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_snapshots(root, cpu_spec2026_int_curves=SPEC_CSV)
            entry = dataset_of(payload_for(root), "SPEC INT")

            self.assertTrue(entry["sparseCurves"])
            self.assertEqual(entry["coreGroups"], ["super", "medium"])
            self.assertEqual(entry["coreNames"], ["Oryon V3 L", "Oryon V3 M"])
            profile = entry["profiles"][0]
            self.assertEqual(profile["label"], "Snapdragon 8 Elite Gen 5 — Oryon V3 L · super · p")
            self.assertEqual(profile["legend"], "Snapdragon 8 Elite Gen 5 — Oryon V3 L")
            self.assertEqual(profile["leader"],
                             "Snapdragon 8 Elite Gen 5 — Oryon V3 L · Super · P-core")
            self.assertEqual(profile["coreGroup"], "super")

    def test_battery_profiles_carry_measured_overlay_and_processor_averages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_snapshots(root, battery_results=BATTERY_CSV)
            entry = dataset_of(payload_for(root), "Battery")

            self.assertEqual(entry["kind"], "battery")
            by_label = {profile["label"]: profile for profile in entry["profiles"]}
            iphone = by_label["苹果 · iPhone 17 Pro Max"]
            self.assertAlmostEqual(iphone["minPerWh"], 600 / 17.5)
            self.assertIsNotNone(iphone["measured"])
            self.assertEqual(iphone["measured"]["measuredMah"], 4718)
            # The adjusted point is a separate overlay, never a replacement.
            self.assertLess(iphone["measured"]["capacityWh"], iphone["capacityWh"])
            self.assertIsNone(by_label["Honor · X70"]["measured"])
            self.assertEqual(by_label["Honor · X70"]["soc"], "")

            socs = {average["soc"]: average for average in entry["socAverages"]}
            self.assertEqual(set(socs), {"Apple A19 Pro", "Snapdragon 8 Elite Gen 5"})
            self.assertEqual(socs["Apple A19 Pro"]["deviceCount"], 1)

    def test_missing_datasets_expose_their_collector_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_snapshots(root, cpu_gb6_curves=CPU_CSV)
            payload = payload_for(root)
            battery = dataset_of(payload, "Battery")

            self.assertFalse(battery["available"])
            self.assertEqual(battery["hintFile"], "snapshots/battery_results.csv")
            self.assertIn("battery_parser.py", battery["hintCommand"])
            # The page reads these unconditionally, so they must still be present.
            self.assertEqual(battery["profiles"], [])
            self.assertEqual(battery["socAverages"], [])
            self.assertEqual(battery["views"], list(DATASET_DEFINITIONS["Battery"].views))

    def test_labels_match_the_tk_dashboard_for_non_spec_datasets(self) -> None:
        self.assertEqual(legend_label("GPU", "A19 Pro"), "A19 Pro")
        self.assertEqual(legend_label("SPEC FP", "A19 Pro — Sawtooth · medium · e"),
                         "A19 Pro — Sawtooth")
        rows = pd.DataFrame({"__label": ["RTX 5090"]})
        self.assertEqual(leader_label("GPU", rows), "RTX 5090")


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        write_snapshots(self.root, cpu_gb6_curves=CPU_CSV)
        self.state = DashboardState(self.root, None, None)
        self.state.refresh()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(DashboardHandler, state=self.state))
        self.addCleanup(self.server.server_close)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def get(self, path: str) -> tuple[int, str, bytes]:
        with urllib.request.urlopen(self.base + path, timeout=10) as response:
            return response.status, response.headers["Content-Type"], response.read()

    def test_serves_the_page_and_its_assets(self) -> None:
        status, content_type, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"SoCPK Comparison Lab", body)
        for asset in ("/app.css", "/chart.js", "/app.js"):
            status, _, body = self.get(asset)
            self.assertEqual(status, 200, asset)
            self.assertTrue(body)

    def test_api_returns_the_payload_as_json(self) -> None:
        status, content_type, body = self.get("/api/data")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        payload = json.loads(body)
        self.assertEqual(dataset_of(payload, "CPU")["pointCount"], 6)

    def test_reload_picks_up_a_snapshot_written_after_startup(self) -> None:
        self.assertFalse(dataset_of(json.loads(self.get("/api/data")[2]), "Battery")["available"])
        write_snapshots(self.root, battery_results=BATTERY_CSV)
        payload = json.loads(self.get("/api/reload")[2])
        self.assertTrue(dataset_of(payload, "Battery")["available"])
        # The cached payload is replaced, not just the response.
        self.assertTrue(dataset_of(self.state.payload, "Battery")["available"])

    def test_refuses_paths_outside_the_web_folder(self) -> None:
        for path in ("/../socpk_web.py", "/%2e%2e/socpk_web.py", "/nope.js"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(path)
            self.assertEqual(caught.exception.code, 404, path)
            caught.exception.close()


class BindingTests(unittest.TestCase):
    """Safari's HTTPS-Only mode rejects plain HTTP, and localhost resolves to
    ::1 first on macOS, so both loopback families have to answer."""

    def serve(self, host: str, https: bool = False):
        state = DashboardState(Path("."), [], None)
        state.refresh()
        servers = build_servers(host, 0, partial(DashboardHandler, state=state))
        if https:
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            wrap_with_tls(servers, *ensure_certificate(Path(directory.name)))
        for server in servers:
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        return servers

    def fetch(self, url: str, insecure: bool = False) -> int:
        context = None
        if insecure:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, timeout=10, context=context) as response:
            return response.status

    def test_display_host_prefers_localhost_over_loopback_literals(self) -> None:
        for host in ("127.0.0.1", "::1", "localhost"):
            self.assertEqual(display_host(host), "localhost")
        self.assertEqual(display_host("192.168.1.5"), "192.168.1.5")

    def test_loopback_binds_both_families_on_one_port(self) -> None:
        servers = self.serve("localhost")
        self.assertEqual(len(servers), 2)
        port = servers[0].server_address[1]
        self.assertEqual(servers[1].server_address[1], port)
        self.assertEqual(self.fetch(f"http://127.0.0.1:{port}/api/data"), 200)
        self.assertEqual(self.fetch(f"http://[::1]:{port}/api/data"), 200)

    def test_an_explicit_interface_binds_only_itself(self) -> None:
        servers = self.serve("127.0.0.1")
        # A literal is a loopback name too, so it still gets both families.
        self.assertEqual(len(servers), 2)

    @unittest.skipUnless(shutil.which("openssl") or shutil.which("mkcert"),
                         "needs openssl or mkcert to build a local certificate")
    def test_https_serves_the_dashboard_over_tls(self) -> None:
        servers = self.serve("localhost", https=True)
        port = servers[0].server_address[1]
        self.assertEqual(self.fetch(f"https://localhost:{port}/api/data", insecure=True), 200)

    @unittest.skipUnless(shutil.which("openssl") or shutil.which("mkcert"),
                         "needs openssl or mkcert to build a local certificate")
    def test_certificate_is_generated_once_and_kept_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            certificate, key = ensure_certificate(Path(directory))
            self.assertTrue(certificate.is_file() and key.is_file())
            self.assertEqual(key.stat().st_mode & 0o777, 0o600)
            stamp = certificate.stat().st_mtime_ns
            self.assertEqual(ensure_certificate(Path(directory))[0], certificate)
            self.assertEqual(certificate.stat().st_mtime_ns, stamp)


class AssetTests(unittest.TestCase):
    def test_page_loads_every_shipped_script_and_stylesheet(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        for asset in ("app.css", "chart.js", "app.js"):
            self.assertTrue((WEB_ROOT / asset).is_file(), asset)
            self.assertIn(asset, index)

    def test_every_benchmark_tab_has_a_label(self) -> None:
        payload = build_payload({}, [], Path("."))
        labels = {entry["key"]: entry["tabLabel"] for entry in payload["datasets"]}
        for benchmark in CPU_BENCHMARKS.values():
            self.assertTrue(labels[benchmark.dataset_key])
        self.assertEqual(labels["Battery"], "BATTERY")


if __name__ == "__main__":
    unittest.main()

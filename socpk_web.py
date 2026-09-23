#!/usr/bin/env python3
"""Serve the SoCPK comparison dashboard as a local web app.

The Tk dashboard in ``socpk_gui.py`` stays as a fallback.  This launcher reads
the same snapshots through ``socpk_data``, converts them to JSON once, and lets
the browser own every interaction, so filtering, hovering and redrawing never
wait on a Python round trip.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import ssl
import subprocess
import threading
import time
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from cpu_benchmarks import CPU_BENCHMARKS
from socpk_data import (
    CURVE_DATASETS,
    DATASET_DEFINITIONS,
    DATASET_SOURCE_HINTS,
    PALETTE,
    SOC_AVERAGE_VIEWS,
    add_soc_average_columns,
    load_collections,
    soc_average_summary,
)

WEB_ROOT = Path(__file__).resolve().parent / "web"
SPEC_DATASETS = frozenset(
    benchmark.dataset_key for benchmark in CPU_BENCHMARKS.values() if benchmark.single_core
)
CORE_GROUP_ORDER = ("super", "large", "medium", "small")
# Names that mean "this machine only". `localhost` resolves to ::1 before
# 127.0.0.1 on macOS, so a loopback bind covers both families or browsers hit a
# refused connection first.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# `.cache` is already ignored by git, so the generated certificate stays out of
# the repository without a new ignore rule.
CERT_DIR = Path(__file__).resolve().parent / ".cache" / "socpk-web"
CERT_DAYS = 365
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
}


def _number(value: Any) -> float | None:
    """Return a JSON-safe float; NaN and infinities become null."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


def legend_label(key: str, label: str) -> str:
    """Drop the core group and variant from a SPEC label, as the Tk legend does."""
    if key in SPEC_DATASETS and " — " in label:
        chip, core = label.split(" — ", 1)
        return f"{chip} — {core.split(' · ', 1)[0]}"
    return label


def leader_label(key: str, rows: pd.DataFrame) -> str:
    """Spell out the core type behind a SPEC profile for the leader card."""
    if key not in SPEC_DATASETS or rows.empty or "Core" not in rows:
        return _text(rows.iloc[0]["__label"]) if not rows.empty else ""
    row = rows.iloc[0]
    parts = [_text(row["Core"]), _text(row["Core_Group"]).capitalize()]
    variant = _text(row.get("Core_Variant", ""))
    if variant:
        parts.append({"p": "P-core", "e": "E-core"}.get(variant, variant))
    return f"{_text(row['CPU'])} — " + " · ".join(part for part in parts if part)


def _curve_dataset(key: str, frame: pd.DataFrame) -> dict[str, Any]:
    definition = DATASET_DEFINITIONS[key]
    score_column = definition.score_column
    is_spec = key in SPEC_DATASETS

    profiles = []
    for label, rows in frame.groupby("__label", sort=False):
        series = []
        for source, source_rows in rows.groupby("__source", sort=False):
            source_rows = source_rows.sort_values("Board_Power_W")
            points = [
                [power, score, efficiency]
                for power, score, efficiency in zip(
                    source_rows["Board_Power_W"].astype(float),
                    source_rows[score_column].astype(float),
                    source_rows["Efficiency"].astype(float),
                )
                if all(math.isfinite(value) for value in (power, score, efficiency))
            ]
            if points:
                series.append({"source": _text(source), "points": points})
        if not series:
            continue
        first = rows.iloc[0]
        profiles.append({
            "label": _text(label),
            "legend": legend_label(key, _text(label)),
            "leader": leader_label(key, rows),
            "core": _text(first.get("Core", "")) if is_spec else "",
            "coreGroup": _text(first.get("Core_Group", "")) if is_spec else "",
            "series": series,
            "count": sum(len(item["points"]) for item in series),
        })

    profiles.sort(key=lambda profile: profile["label"].casefold())
    core_names = sorted({profile["core"] for profile in profiles if profile["core"]}, key=str.casefold)
    present_groups = {profile["coreGroup"] for profile in profiles if profile["coreGroup"]}
    return {
        "kind": "curve",
        "scoreLabel": definition.score_label,
        "scoreDecimals": definition.score_decimals,
        "sparseCurves": is_spec,
        "coreGroups": [group for group in CORE_GROUP_ORDER if group in present_groups],
        "coreNames": core_names,
        "profiles": profiles,
        "pointCount": sum(profile["count"] for profile in profiles),
    }


def _battery_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    from battery_metadata import review_record, brand_name
    frame = pd.DataFrame([review_record(row) for row in frame.to_dict("records")])
    frame = add_soc_average_columns(frame)
    summary = frame.groupby("__label", as_index=False).agg(
        minutes=("minutes", "mean"),
        hours=("hours", "mean"),
        capacityWh=("capacityWh", "mean"),
        avgPowerW=("avgPowerW", "mean"),
        minPerWh=("minPerWh", "mean"),
        geekerwanAdvertisedMah=("geekerwanAdvertisedMah", "mean"),
        geekerwanMeasuredMah=("geekerwanMeasuredMah", "mean"),
        geekerwanShortfallMah=("geekerwanShortfallMah", "mean"),
        geekerwanShortfallPct=("geekerwanShortfallPct", "mean"),
        geekerwanCapacityWh=("geekerwanCapacityWh", "mean"),
        geekerwanAvgPowerW=("geekerwanAvgPowerW", "mean"),
        geekerwanMinPerWh=("geekerwanMinPerWh", "mean"),
        soc=("soc", "first"),
        rows=("minutes", "size"),
    )

    profiles = []
    for _, row in summary.iterrows():
        source_rows = frame[frame["__label"].eq(row["__label"])]
        first = source_rows.iloc[0]
        measured = None
        if _number(row["geekerwanMeasuredMah"]) is not None:
            measured = {
                "advertisedMah": _number(row["geekerwanAdvertisedMah"]),
                "measuredMah": _number(row["geekerwanMeasuredMah"]),
                "shortfallMah": _number(row["geekerwanShortfallMah"]),
                "shortfallPct": _number(row["geekerwanShortfallPct"]),
                "capacityWh": _number(row["geekerwanCapacityWh"]),
                "avgPowerW": _number(row["geekerwanAvgPowerW"]),
                "minPerWh": _number(row["geekerwanMinPerWh"]),
            }
        profiles.append({
            "label": _text(row["__label"]),
            "legend": _text(row["__label"]),
            "leader": _text(row["__label"]),
            "soc": _text(row["soc"]),
            "brand": brand_name(first.get("brand", "")),
            "model": _text(first.get("model", "")),
            "os": _text(first.get("os", "")),
            "screenSize": _number(first.get("screen_size_in")),
            "refreshHz": _number(first.get("refresh_hz")),
            "sources": sorted(source_rows["__source"].unique()),
            "metadataReview": _text(first.get("metadata_review", "")),
            "metadataSource": _text(first.get("metadata_source", "")),
            "metadataReviewed": _text(first.get("metadata_reviewed", "")),
            "minutes": _number(row["minutes"]),
            "hours": _number(row["hours"]),
            "capacityWh": _number(row["capacityWh"]),
            "avgPowerW": _number(row["avgPowerW"]),
            "minPerWh": _number(row["minPerWh"]),
            "measured": measured,
            "count": int(row["rows"]),
        })
    profiles.sort(key=lambda profile: profile["label"].casefold())

    averages = soc_average_summary(frame)
    soc_averages = [
        {
            "soc": _text(row["soc"]),
            "deviceCount": int(row["soc_device_count"]) if _number(row["soc_device_count"]) else 0,
            "capacityWh": _number(row["soc_avg_capacity_wh"]),
            "powerW": _number(row["soc_avg_power_w"]),
            "minPerWh": _number(row["soc_avg_min_per_wh"]),
            "runtimeHours": _number(row["soc_avg_runtime_hours"]),
        }
        for _, row in averages.iterrows()
    ]
    return {
        "kind": "battery",
        "scoreDecimals": 0,
        "sparseCurves": False,
        "coreGroups": [],
        "coreNames": [],
        "profiles": profiles,
        "socAverages": soc_averages,
        "socViews": sorted(SOC_AVERAGE_VIEWS),
        "pointCount": len(frame),
    }


def build_payload(
    collections: dict[str, pd.DataFrame],
    warnings: list[str],
    root: Path,
    initial_dataset: str | None = None,
) -> dict[str, Any]:
    """Convert the loaded snapshots into the single JSON document the page reads."""
    datasets = []
    for key, definition in DATASET_DEFINITIONS.items():
        hint_file, hint_command = DATASET_SOURCE_HINTS[key]
        entry: dict[str, Any] = {
            "key": key,
            "tabLabel": definition.tab_label or key.upper(),
            "title": definition.title,
            "kicker": definition.kicker,
            "views": list(definition.views),
            "hintFile": hint_file,
            "hintCommand": hint_command,
            "available": key in collections,
            "sourceCount": 0,
            "profiles": [],
            "pointCount": 0,
            "kind": "curve" if key in CURVE_DATASETS else "battery",
            "coreGroups": [],
            "coreNames": [],
            "scoreDecimals": definition.score_decimals,
            "scoreLabel": definition.score_label,
            "sparseCurves": False,
            "socAverages": [],
        }
        frame = collections.get(key)
        if frame is not None and not frame.empty:
            entry.update(
                _curve_dataset(key, frame) if key in CURVE_DATASETS else _battery_dataset(frame)
            )
            entry["sourceCount"] = int(frame["__source"].nunique())
        datasets.append(entry)

    available = [entry["key"] for entry in datasets if entry["available"]]
    if initial_dataset not in DATASET_DEFINITIONS:
        initial_dataset = available[0] if available else datasets[0]["key"]
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "initialDataset": initial_dataset,
        "warnings": list(warnings),
        "palette": list(PALETTE),
        "datasets": datasets,
    }


class DashboardState:
    """Hold the newest payload and rebuild it when the page asks for a reload."""

    def __init__(self, root: Path, csv_paths: list[Path] | None, initial_dataset: str | None):
        self.root = root
        self.csv_paths = csv_paths
        self.initial_dataset = initial_dataset
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = {}

    def refresh(self) -> dict[str, Any]:
        collections, warnings = load_collections(self.root, self.csv_paths)
        payload = build_payload(collections, warnings, self.root, self.initial_dataset)
        with self._lock:
            self._payload = payload
        return payload

    @property
    def payload(self) -> dict[str, Any]:
        with self._lock:
            return self._payload or self.refresh()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SoCPKWeb/1.0"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, state: DashboardState, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *_args) -> None:  # quiet; the console belongs to the user
        pass

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if route == "/api/data":
            self._send_json(self.state.payload)
        elif route == "/api/reload":
            self._send_json(self.state.refresh())
        else:
            self._send_static(route)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] == "/api/reload":
            self._send_json(self.state.refresh())
        else:
            self.send_error(404)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, route: str) -> None:
        relative = "index.html" if route in {"", "/"} else route.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        # Keep path traversal out of the served tree.
        if not target.is_file() or WEB_ROOT.resolve() not in target.parents:
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def ensure_certificate(directory: Path = CERT_DIR) -> tuple[Path, Path]:
    """Return a certificate and key for loopback, generating them once.

    mkcert produces one the system already trusts, so the browser loads the
    page with no interstitial.  Without it, openssl produces a self-signed
    certificate that needs a one-time click-through per browser.
    """
    certificate = directory / "loopback-cert.pem"
    key = directory / "loopback-key.pem"
    fresh = (
        certificate.is_file()
        and key.is_file()
        and time.time() - certificate.stat().st_mtime < CERT_DAYS * 0.9 * 86400
    )
    if fresh:
        return certificate, key

    directory.mkdir(parents=True, exist_ok=True)
    mkcert = shutil.which("mkcert")
    if mkcert:
        command = [mkcert, "-cert-file", str(certificate), "-key-file", str(key),
                   "localhost", "127.0.0.1", "::1"]
    else:
        openssl = shutil.which("openssl")
        if not openssl:
            raise SystemExit(
                "--https needs mkcert or openssl on PATH to create a local certificate."
            )
        command = [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", str(CERT_DAYS),
            "-keyout", str(key), "-out", str(certificate), "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1",
        ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not certificate.is_file():
        raise SystemExit(f"Could not create a local certificate:\n{result.stderr.strip()}")
    key.chmod(0o600)
    return certificate, key


def wrap_with_tls(servers: list[ThreadingHTTPServer], certificate: Path, key: Path) -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    for server in servers:
        server.socket = context.wrap_socket(server.socket, server_side=True)


def build_servers(host: str, port: int, handler) -> list[ThreadingHTTPServer]:
    """Listen on every loopback family, so localhost and 127.0.0.1 both work."""
    if host not in LOOPBACK_HOSTS:
        return [ThreadingHTTPServer((host, port), handler)]

    servers = [ThreadingHTTPServer(("127.0.0.1", port), handler)]
    # --port 0 picked a port; mirror that exact port on IPv6 loopback.
    port = servers[0].server_address[1]
    try:
        servers.append(_IPv6Server(("::1", port), handler))
    except OSError:
        pass  # No IPv6 loopback here; 127.0.0.1 still answers localhost.
    return servers


def display_host(host: str) -> str:
    """Safari's HTTPS-Only mode rejects loopback literals but allows localhost."""
    return "localhost" if host in LOOPBACK_HOSTS else host


def open_browser(url: str, name: str | None) -> None:
    try:
        browser = webbrowser.get(name) if name else webbrowser.get()
    except webbrowser.Error:
        print(f"  no browser named {name!r}; open {url} yourself", flush=True)
        return
    browser.open(url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the SoCPK comparison dashboard in a browser")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project folder to scan for CSV files")
    parser.add_argument("--csv", "--input", dest="csv_paths", type=Path, action="append",
                        help="Load only this CSV; repeat to load multiple files")
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFINITIONS),
                        help="Dataset shown at startup")
    parser.add_argument("--host", default="localhost",
                        help="Interface to bind; loopback names bind IPv4 and IPv6")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind; 0 picks a free one")
    parser.add_argument("--browser", help="Open this browser instead of the default, e.g. chrome")
    parser.add_argument("--https", action="store_true",
                        help="Serve over TLS with a local certificate, for browsers that "
                             "refuse plain HTTP (Safari's HTTPS-Only mode)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    paths = [path.resolve() for path in args.csv_paths] if args.csv_paths else None
    missing = [path for path in paths or [] if not path.is_file()]
    if missing:
        raise SystemExit(f"CSV file does not exist: {missing[0]}")

    state = DashboardState(root, paths, args.dataset)
    payload = state.refresh()
    loaded = sum(1 for entry in payload["datasets"] if entry["available"])
    servers = build_servers(args.host, args.port, partial(DashboardHandler, state=state))
    port = servers[0].server_address[1]
    scheme = "http"
    if args.https:
        certificate, key = ensure_certificate()
        wrap_with_tls(servers, certificate, key)
        scheme = "https"
    url = f"{scheme}://{display_host(args.host)}:{port}/"
    print(f"SoCPK dashboard on {url}", flush=True)
    if args.https and not shutil.which("mkcert"):
        print("  self-signed certificate: accept it once per browser "
              "(install mkcert to skip this)", flush=True)
    print(f"Scanning {root} · {loaded} dataset(s) loaded", flush=True)
    for warning in payload["warnings"]:
        print(f"  warning: {warning}", flush=True)
    if not args.no_browser:
        threading.Timer(0.35, open_browser, (url, args.browser)).start()

    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        while True:
            threads[0].join(timeout=3600)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()

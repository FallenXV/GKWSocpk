#!/usr/bin/env python3
"""Fast sanity check: sources compile, the dashboard serves, the data API answers.

This is the only verification step agents should run. It writes nothing to disk.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import urllib.request
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
SKIP = {".venv", ".git", "__pycache__", ".cache", ".gsm_cache", "snapshots", "tests"}


def sources(suffix):
    return [p for p in ROOT.rglob(f"*{suffix}") if not SKIP.intersection(p.relative_to(ROOT).parts)]


def main() -> int:
    failures = []

    for path in sources(".py"):
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as error:
            failures.append(f"python syntax: {path.relative_to(ROOT)}:{error.lineno}: {error.msg}")

    node = shutil.which("node")
    for path in sources(".js") if node else []:
        result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        if result.returncode:
            failures.append(f"js syntax: {path.relative_to(ROOT)}\n{result.stderr.strip()}")

    import socpk_web

    state = socpk_web.DashboardState(ROOT, None, None)
    state.refresh()
    server = socpk_web.ThreadingHTTPServer(("127.0.0.1", 0), partial(socpk_web.DashboardHandler, state=state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for route in ("/", "/app.css", "/colors.js", "/chart.js", "/analytics.js", "/app.js"):
            with urllib.request.urlopen(base + route, timeout=10) as response:
                if response.status != 200 or not response.read():
                    failures.append(f"static {route}: HTTP {response.status}")
        for method, route in (("GET", "/api/data"), ("POST", "/api/reload")):
            request = urllib.request.Request(base + route, method=method)
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if not payload.get("datasets"):
                failures.append(f"{method} {route}: no datasets in payload")
        loaded = [d["key"] for d in payload["datasets"] if d["available"]]
        for warning in payload.get("warnings", []):
            print(f"warning: {warning}")
        print(f"datasets loaded: {', '.join(loaded) or 'none'}")
    except Exception as error:  # report, don't traceback-dump
        failures.append(f"server: {error!r}")
    finally:
        server.shutdown()
        server.server_close()

    for failure in failures:
        print(f"FAIL {failure}")
    print("SMOKE FAILED" if failures else "SMOKE OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

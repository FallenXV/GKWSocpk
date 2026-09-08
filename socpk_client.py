"""Read SoCPK's public chart API and explicitly supplied legacy assets."""

from __future__ import annotations

import base64
import json
import math
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from xml.etree import ElementTree

import requests

SOCPK_ROOT = "https://www.socpk.com/"
RANKINGS_PAYLOAD_KEY = "socpk-rankings-2026"
CURVES_PAYLOAD_KEY = "socpk-curves-2026"

CPU_PAGE_SLUG = "mobile-soc-efficiency-gb7"
GPU_PAGE_SLUG = "mobile-soc-efficiency-snl"
LAPTOP_GPU_PAGE_SLUG = "laptop-gpu-efficiency"
BATTERY_PAGE_SLUG = "battery-life-5-0"

_USER_AGENT = "Mozilla/5.0 (SoCPK parser)"


def decode_chart_blob(encoded: str) -> float | list:
    """Decode the public API's chart codec (September 2026).

    Python equivalent of the site's WebAssembly decoder: seeded Mulberry32,
    unsigned/zigzag varints, delta-coded line coordinates and affine scaling.
    Values are already in chart units; no SVG axis conversion is needed.
    """
    raw = base64.b64decode(encoded, validate=True)
    if not 6 <= len(raw) <= 262144 or raw[0] not in (1, 2):
        raise ValueError("Unsupported or invalid SoCPK chart blob.")
    state = int.from_bytes(raw[1:5], "little") ^ 0x9E3779B9
    offset = 5

    def random() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        value = ((state ^ (state >> 15)) * (state | 1)) & 0xFFFFFFFF
        value ^= (value + ((value ^ (value >> 7)) * (value | 61))) & 0xFFFFFFFF
        return (value ^ (value >> 14)) / 4294967296.0

    def varint() -> int:
        nonlocal offset
        value = 0
        for shift in range(0, 56, 7):
            if offset >= len(raw):
                break
            byte = raw[offset]
            offset += 1
            value |= (byte & 127) << shift
            if not byte & 128:
                if value > 2**53 - 1:
                    break
                return value
        raise ValueError("Truncated or oversized SoCPK chart varint.")

    def signed() -> int:
        value = varint()
        return (value >> 1) ^ -(value & 1)

    is_line = raw[0] == 1
    y_scale, y_offset = random() * 1.8 + 0.6, (random() * 2 - 1) * 5000
    x_scale, x_offset = random() * 1.8 + 0.6, (random() * 2 - 1) * 5000
    count = varint()
    if count > (65536 if is_line else 131072):
        raise ValueError("Too many values in SoCPK chart blob.")
    values = []
    x = y = 0
    for _ in range(count):
        if is_line:
            x_noise, y_noise = (random() * 2 - 1) * 2500, (random() * 2 - 1) * 2500
            x += signed()
            y += signed()
            values.append([
                (x / 10000 - x_offset - x_noise) / x_scale,
                (y / 10000 - y_offset - y_noise) / y_scale,
            ])
        else:
            noise = (random() * 2 - 1) * 2500
            values.append((signed() / 10000 - y_offset - noise) / y_scale)
    if offset != len(raw):
        raise ValueError("Unexpected trailing SoCPK chart data.")
    return values[0] if not is_line and len(values) == 1 else values


def fetch_chart_page(
    slug: str, root_url: str = SOCPK_ROOT, timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch metadata and public chart data, refreshing stale tokens once.

    Tokens and data are deliberately not cached between polls. The two GETs
    follow the site's own client, including its data-version consistency check.
    """
    page_url = urljoin(root_url, f"/api/pages/{slug}")
    with requests.Session() as session:
        session.headers.update({"User-Agent": _USER_AGENT})
        for attempt in range(2):
            params = {"_": str(time.time_ns() // 1000000)} if attempt else None
            response = session.get(page_url, params=params, timeout=timeout)
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, dict) or not isinstance(page.get("series"), list):
                raise ValueError(f"Invalid SoCPK page: {slug}.")
            if not page.get("dataToken") or not page.get("dataExp"):
                if all("points" in series for series in page["series"]):
                    return page
                raise ValueError(f"Missing SoCPK chart token: {slug}.")
            if attempt == 0 and page["dataExp"] <= time.time() + 10:
                continue
            response = session.get(
                page_url + "/data", params=params, timeout=timeout,
                headers={"X-Chart-Token": f"{page['dataExp']}.{page['dataToken']}"},
            )
            if response.status_code == 403 and attempt == 0:
                continue
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or data.get("v") != 1:
                raise ValueError("Unsupported SoCPK chart response version.")
            if page.get("dataVersion") != data.get("dataVersion"):
                if attempt == 0:
                    continue
                raise ValueError(f"SoCPK chart changed during fetch: {slug}.")
            if not isinstance(data.get("series"), list):
                raise ValueError("Missing SoCPK chart series.")
            blobs = {series["id"]: series["p"] for series in data["series"]}
            decoded = []
            for series in page["series"]:
                if series["id"] not in blobs:
                    raise ValueError(f"Missing SoCPK data for series {series['id']}.")
                decoded.append({**series, "points": decode_chart_blob(blobs[series["id"]])})
            return {**page, "series": decoded}
    raise ValueError(f"Could not refresh SoCPK chart data: {slug}.")


_WHITESPACE_RE = re.compile(r"\s+")
_NAME_PREFIXES = tuple(
    (re.compile(rf"{short}\d"), len(short), full)
    for short, full in (("sd", "snapdragon"), ("d", "dimensity"),
                        ("k", "kirin"), ("e", "exynos"))
)


@lru_cache(maxsize=1024)
def _curve_name_key(name: str) -> str:
    """Accept previous CLI abbreviations without conflating RAM variants."""
    name = _WHITESPACE_RE.sub("", name).casefold().removesuffix("bionic")
    for pattern, length, full in _NAME_PREFIXES:
        if pattern.match(name):
            return full + name[length:]
    return name


def fetch_curve_series(
    slug: str, names=None, *, suite: str | None = None,
) -> list[dict[str, Any]]:
    """Get validated series from one fresh poll, optionally selecting a suite."""
    return select_curve_series(fetch_chart_page(slug), names, suite=suite)


def select_curve_series(
    page: dict[str, Any], names=None, *, suite: str | None = None,
    multiple_cores: bool = False,
) -> list[dict[str, Any]]:
    """Select and validate curves; SPEC chip selections include all its cores."""
    if page.get("type") != "line-chart" or page.get("config", {}).get("xUnit") != "W":
        raise ValueError("Expected a SoCPK power-versus-score chart in watts.")
    series = [item for item in page["series"]
              if suite is None or item.get("meta", {}).get("suite") == suite]
    if names is not None:
        by_key: dict[str, list[int]] = {}
        for index, item in enumerate(series):
            for key in {_curve_name_key(item.get("name", "")),
                        _curve_name_key(item.get("name_en", ""))}:
                by_key.setdefault(key, []).append(index)
        selected: list[dict[str, Any]] = []
        chosen: set[int] = set()
        for name in names:
            matches = by_key.get(_curve_name_key(name), ())
            if not matches or (len(matches) != 1 and not multiple_cores):
                raise ValueError(f"Unknown or ambiguous SoCPK curve: {name!r}.")
            for index in matches:
                if index not in chosen:
                    chosen.add(index)
                    selected.append(series[index])
        series = selected
    for item in series:
        points = item.get("points")
        if not isinstance(points, list):
            raise ValueError(f"Invalid points for {item['name']}.")
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
                    or point[0] <= 0 or point[1] < 0):
                raise ValueError(f"Invalid power/score point for {item['name']}.")
    return series


def series_label(item: dict[str, Any]) -> str:
    """Prefer the published English name, falling back to the local name."""
    return item.get("name_en") or item["name"]


def curve_frame(series, id_column: str | None, score_column: str):
    """Convert decoded coordinates into the existing CSV schema.

    ``id_column`` carries the chip name; ``None`` omits it for a single curve.
    """
    import pandas as pd

    labels, powers, scores = [], [], []
    for item in series:
        label = series_label(item)
        for power, score in item["points"]:
            labels.append(label)
            powers.append(power)
            scores.append(score)
    columns = {"Board_Power_W": powers, score_column: scores,
               "Efficiency": [score / power for power, score in zip(powers, scores)]}
    if id_column:
        columns = {id_column: labels, **columns}
    return pd.DataFrame(columns)


def battery_rows_from_page(page: dict[str, Any]) -> list[list]:
    """Adapt API battery metadata and runtime to the legacy seven-column rows."""
    if page.get("type") != "bar-chart" or page.get("config", {}).get("unit") != "min":
        raise ValueError("Expected a SoCPK battery runtime chart in minutes.")
    rows = []
    for series in page["series"]:
        meta = series.get("meta", {})
        minutes = series.get("points")
        if isinstance(minutes, list):
            minutes = minutes[0] if minutes else None
        capacity = meta.get("ratedEnergyWh")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                   for v in (minutes, capacity)):
            continue
        brand = series.get("group", "")
        model = series["name"]
        if brand and model.casefold().startswith(brand.casefold() + " "):
            model = model[len(brand):].strip()
        # Remove codec quantization noise from runtime in minutes.
        rows.append([brand, model, round(minutes, 3), meta.get("systemVersion", ""), "",
                     capacity, meta.get("videoUrl", "")])
    if not rows:
        raise ValueError("SoCPK API contains no usable battery results.")
    return rows


@contextmanager
def new_snapshot(path: str | Path):
    """Exclusively create an export; existing files always remain untouched.

    A timestamped sibling is created when the requested path already exists.
    Exclusive creation also protects against concurrent polls and collisions.
    """
    requested = Path(path)
    requested.parent.mkdir(parents=True, exist_ok=True)
    candidate = requested
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sequence = 0
    while True:
        try:
            handle = candidate.open("x", newline="", encoding="utf-8")
            break
        except FileExistsError:
            sequence += 1
            candidate = requested.with_name(
                f"{requested.stem}_{stamp}_{sequence}{requested.suffix}"
            )
    try:
        with handle:
            yield handle
    except BaseException:
        candidate.unlink()
        raise

_MODULE_RE = re.compile(
    r"<script\b[^>]*\bsrc=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PATH_TOKEN_RE = re.compile(rf"[A-Za-z]|{_NUMBER}")


def discover_module_urls(html: str, page_url: str) -> list[str]:
    """Return absolute module-script URLs referenced by a SoCPK HTML page."""
    return [urljoin(page_url, src) for src in _MODULE_RE.findall(html)]


def decode_embedded_payload(bundle_text: str, key: str) -> Any:
    """Decode a base64/XOR JSON payload embedded in the application bundle."""
    key_prefix, separator, key_suffix = key.rpartition("-")
    if separator and key_suffix.isdigit():
        accepted_key = rf"{re.escape(key_prefix)}-[^`]+"
    else:
        accepted_key = re.escape(key)
    pattern = re.compile(
        rf"[A-Za-z_$][\w$]*\(\s*`(?P<data>[A-Za-z0-9+/=]+)`\s*,\s*"
        rf"`(?P<key>{accepted_key})`\s*\)"
    )
    match = pattern.search(bundle_text)
    if not match:
        raise ValueError(f"Could not find embedded SoCPK payload for key {key!r}.")

    try:
        encrypted = base64.b64decode(match.group("data"), validate=True)
        actual_key = match.group("key")
        key_bytes = actual_key.encode("utf-8")
        decoded = bytes(
            byte ^ key_bytes[index % len(key_bytes)]
            for index, byte in enumerate(encrypted)
        )
        return json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid SoCPK payload for key {key!r}.") from exc


@lru_cache(maxsize=8)
def fetch_spa_payload(
    key: str,
    page_url: str = SOCPK_ROOT,
    timeout: float = 30.0,
) -> Any:
    """Fetch a SoCPK page, discover its versioned bundle, and decode a payload."""
    headers = {"User-Agent": _USER_AGENT}
    page = requests.get(page_url, headers=headers, timeout=timeout)
    page.raise_for_status()

    module_urls = discover_module_urls(page.text, page.url)
    if not module_urls:
        raise ValueError(f"No module script found at {page.url}.")

    errors: list[str] = []
    for module_url in module_urls:
        try:
            bundle = requests.get(module_url, headers=headers, timeout=timeout)
            bundle.raise_for_status()
            return decode_embedded_payload(bundle.text, key)
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{module_url}: {exc}")

    detail = "; ".join(errors)
    raise ValueError(f"Could not decode SoCPK payload. {detail}")


def fetch_curve_manifest(page_url: str = SOCPK_ROOT) -> dict[str, Any]:
    """Read a legacy embedded curve manifest from an explicitly supplied SPA."""
    payload = fetch_spa_payload(CURVES_PAYLOAD_KEY, page_url)
    if not isinstance(payload, dict):
        raise ValueError("SoCPK curve payload is not an object.")
    return payload


def extract_axis_geometry(
    svg: str,
    *,
    power_range: float,
    score_range: float,
) -> dict[str, float] | None:
    """Extract plot geometry from current or legacy SoCPK axis SVG markup."""
    horizontal_re = re.compile(
        rf"M\s*({_NUMBER})[\s,]+({_NUMBER})\s*h\s*({_NUMBER})",
        re.IGNORECASE,
    )
    vertical_re = re.compile(
        rf"M\s*({_NUMBER})[\s,]+({_NUMBER})\s*V\s*({_NUMBER})",
        re.IGNORECASE,
    )
    horizontal = [tuple(map(float, match.groups())) for match in horizontal_re.finditer(svg)]
    vertical = [tuple(map(float, match.groups())) for match in vertical_re.finditer(svg)]

    candidates = []
    for x_start, y_base, x_width in horizontal:
        for vertical_x, vertical_y, y_top in vertical:
            if (
                abs(x_start - vertical_x) < 0.1
                and abs(y_base - vertical_y) < 0.1
                and x_width > 100
                and abs(y_base - y_top) > 100
            ):
                candidates.append(
                    (x_start, y_base, x_width, abs(y_base - y_top))
                )
    if not candidates:
        return None

    # Prefer the inset plotting area over any full-canvas background path.
    x_start, y_base, x_width, y_height = max(
        candidates,
        key=lambda item: (item[0] > 0, item[0], item[2] * item[3]),
    )
    return {
        "X_START": x_start,
        "X_WIDTH": x_width,
        "POWER_RANGE": float(power_range),
        "Y_BASE": y_base,
        "Y_HEIGHT": y_height,
        "SCORE_RANGE": float(score_range),
    }


def extract_curve_coordinates(svg: str) -> list[tuple[float, float]]:
    """Extract line or scatter coordinates from a Matplotlib SoCPK SVG."""
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError:
        return []

    for group in root.iter():
        if group.tag.rsplit("}", 1)[-1] != "g" or group.get("id") != "line2d_1":
            continue
        for element in group.iter():
            if element.tag.rsplit("}", 1)[-1] != "path":
                continue
            path_data = element.get("d", "")
            points = [
                (float(match.group(1)), float(match.group(2)))
                for match in re.finditer(
                    rf"[ML]\s*({_NUMBER})[\s,]+({_NUMBER})",
                    path_data,
                    re.IGNORECASE,
                )
            ]
            if points:
                return points

    points = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "use":
            continue
        try:
            points.append((float(element.attrib["x"]), float(element.attrib["y"])))
        except (KeyError, ValueError):
            continue
    return points


def extract_laptop_curve_coordinates(svg: str) -> list[tuple[float, float]]:
    """Extract a curve from the older Excel-exported laptop GPU SVG format."""
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError:
        return []

    translated_paths = _paths_with_translation(root)
    if not translated_paths:
        return []
    path, translate_x, translate_y = translated_paths[0]
    path_points = _parse_svg_path_points(path.get("d", ""))
    if not path_points:
        return []

    fill = (path.get("fill") or "").strip().lower()
    if fill and fill != "none":
        xs = [point[0] for point in path_points]
        ys = [point[1] for point in path_points]
        path_points = [((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)]

    return [
        (x + translate_x, y + translate_y)
        for x, y in path_points
    ]


def _paths_with_translation(
    root: ElementTree.Element,
) -> list[tuple[ElementTree.Element, float, float]]:
    result = []

    def visit(element: ElementTree.Element, parent_x: float, parent_y: float) -> None:
        translate_x, translate_y = _translation(element.get("transform", ""))
        current_x = parent_x + translate_x
        current_y = parent_y + translate_y
        if (
            element.tag.rsplit("}", 1)[-1] == "path"
            and element.get("d")
        ):
            result.append((element, current_x, current_y))
        for child in element:
            visit(child, current_x, current_y)

    visit(root, 0.0, 0.0)
    return result


def _translation(transform: str) -> tuple[float, float]:
    match = re.search(
        rf"translate\(\s*({_NUMBER})(?:[\s,]+({_NUMBER}))?\s*\)",
        transform,
        re.IGNORECASE,
    )
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2) or 0)


def _parse_svg_path_points(path_data: str) -> list[tuple[float, float]]:
    tokens = _PATH_TOKEN_RE.findall(path_data)
    points = []
    index = 0
    command = ""
    current_x = current_y = 0.0
    start_x = start_y = 0.0

    parameter_counts = {
        "M": 2, "L": 2, "T": 2,
        "H": 1, "V": 1,
        "C": 6, "S": 4, "Q": 4,
        "A": 7,
    }

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command.upper() == "Z":
                current_x, current_y = start_x, start_y
                continue
        if not command:
            return []

        upper = command.upper()
        count = parameter_counts.get(upper)
        if count is None or index + count > len(tokens):
            break
        try:
            values = [float(value) for value in tokens[index:index + count]]
        except ValueError:
            break
        index += count

        relative = command.islower()
        if upper in {"M", "L", "T"}:
            x, y = values[-2:]
        elif upper == "H":
            x, y = values[0], current_y
        elif upper == "V":
            x, y = current_x, values[0]
        else:
            x, y = values[-2:]

        if relative:
            if upper == "H":
                x += current_x
            elif upper == "V":
                y += current_y
            else:
                x += current_x
                y += current_y
        current_x, current_y = x, y
        points.append((x, y))

        if upper == "M":
            start_x, start_y = x, y
            command = "l" if relative else "L"

    return points


def extract_laptop_axis_geometry(svg: str) -> dict[str, float] | None:
    """Extract rendered plot bounds and ranges from laptop GPU axis SVG."""
    translate_x, translate_y = _translation(svg)
    segment_re = re.compile(
        rf"M\s*({_NUMBER})[\s,]+({_NUMBER})[\s,]+"
        rf"({_NUMBER})[\s,]+({_NUMBER})"
    )
    horizontal = []
    vertical = []
    for match in segment_re.finditer(svg):
        x1, y1, x2, y2 = map(float, match.groups())
        if abs(y1 - y2) < 0.1 and abs(x2 - x1) > 100:
            horizontal.append((x1, y1, x2, y2))
        if abs(x1 - x2) < 0.1 and abs(y2 - y1) > 100:
            vertical.append((x1, y1, x2, y2))
    if not horizontal or not vertical:
        return None

    x_start = min(min(segment[0], segment[2]) for segment in horizontal)
    x_end = max(max(segment[0], segment[2]) for segment in horizontal)
    plot_vertical = [
        segment
        for segment in vertical
        if x_start - 1 <= segment[0] <= x_end + 1
    ]
    if not plot_vertical:
        return None
    y_base = max(max(segment[1], segment[3]) for segment in plot_vertical)
    y_top = min(min(segment[1], segment[3]) for segment in plot_vertical)

    labels = [
        float(value)
        for value in re.findall(r"<text\b[^>]*>\s*([0-9.]+)\s*</text>", svg)
    ]
    power_labels = [value for value in labels if value <= 1000]
    score_labels = [value for value in labels if value > 1000]
    if not power_labels or not score_labels:
        return None

    return {
        "X_START": x_start + translate_x,
        "X_WIDTH": x_end - x_start,
        "POWER_RANGE": max(power_labels),
        "Y_BASE": y_base + translate_y,
        "Y_HEIGHT": y_base - y_top,
        "SCORE_RANGE": max(score_labels),
    }

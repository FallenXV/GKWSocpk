"""Helpers for reading data embedded in the current SoCPK single-page app."""

from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from typing import Any
from urllib.parse import urljoin
from xml.etree import ElementTree

import requests

SOCPK_ROOT = "https://www.socpk.com/"
RANKINGS_PAYLOAD_KEY = "socpk-rankings-2026"
CURVES_PAYLOAD_KEY = "socpk-curves-2026"

_MODULE_RE = re.compile(
    r"<script\b[^>]*\bsrc=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


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
    headers = {"User-Agent": "Mozilla/5.0 (SoCPK parser)"}
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


def fetch_rankings(page_url: str = SOCPK_ROOT) -> dict[str, Any]:
    """Return current ranking datasets from SoCPK."""
    payload = fetch_spa_payload(RANKINGS_PAYLOAD_KEY, page_url)
    if not isinstance(payload, dict):
        raise ValueError("SoCPK rankings payload is not an object.")
    return payload


def fetch_curve_manifest(page_url: str = SOCPK_ROOT) -> dict[str, Any]:
    """Return current curve configuration and asset manifest from SoCPK."""
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

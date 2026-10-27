"""Charging chart parsing and shared dashboard metrics.

SoCPK measures input energy at the power supply, including circuit losses.
It is neither stored battery energy nor state of charge.
"""

from __future__ import annotations

import math

from socpk_client import series_label


CHARGING_PAGE_SLUG = "charge-curves"
CHARGING_COLUMNS = (
    "chargingSeries", "model", "brand", "chargingMinutes", "chargedEnergyWh",
    "chargerPowerW", "fullDurationMinutes", "totalEnergyWh",
    "averageChargingPowerW", "soc", "sourceUrl",
)
CHARGING_VIEWS = ("Charged energy", "Charging power", "Charging duration")
CHARGING_METRICS = {
    "Charged energy": ("averagePowerW", "Average input power (W)", True),
    "Charging power": ("peakPowerW", "Peak interval power (W)", True),
    "Charging duration": ("durationMinutes", "Full charge time (min)", False),
}


def positive_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def charging_points(points):
    """Validate and order coordinates, allowing only codec-scale roundoff."""
    if not isinstance(points, list) or not points:
        raise ValueError("Charging series contains no points.")
    ordered = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Invalid charging time/energy point.")
        minute, energy = point
        # This runs twice per loaded sample; avoid generator allocation for
        # the fixed pair while retaining the same validation and rounding.
        if (not isinstance(minute, (int, float)) or isinstance(minute, bool)
                or not math.isfinite(minute) or minute < -0.001
                or not isinstance(energy, (int, float)) or isinstance(energy, bool)
                or not math.isfinite(energy) or energy < -0.001):
            raise ValueError("Invalid charging time/energy point.")
        ordered.append((max(0.0, round(float(minute), 3)),
                        max(0.0, round(float(energy), 3))))
    result = []
    for minute, energy in sorted(ordered):
        if result and minute == result[-1][0]:
            if abs(energy - result[-1][1]) > 0.001001:
                raise ValueError("Conflicting energy at the same charging time.")
            result[-1][1] = max(energy, result[-1][1])
            continue
        if result:
            if energy < result[-1][1] - 0.001001:
                raise ValueError("Cumulative charged energy decreases.")
            energy = max(energy, result[-1][1])
        result.append([minute, energy])
    return result


def charging_rows_from_page(page, names=None):
    """Parse the public minutes/Wh chart without CPU/GPU power-axis assumptions."""
    config = page.get("config", {})
    if (page.get("type") != "line-chart" or config.get("xUnit") != "min"
            or config.get("yUnit") != "Wh"):
        raise ValueError("Expected a SoCPK charging chart in minutes and Wh.")
    series = page.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError("SoCPK charging chart contains no series.")
    if names is not None:
        selected = []
        for name in names:
            matches = [item for item in series if str(name).strip().casefold() in
                       {str(item.get(key, "")).strip().casefold()
                        for key in ("name", "name_en", "id")}]
            if len(matches) != 1:
                raise ValueError(f"Unknown or ambiguous charging device: {name!r}.")
            if matches[0] not in selected:
                selected.append(matches[0])
        series = selected
    rows = []
    for item in series:
        label = series_label(item)
        meta = item.get("meta") or {}
        try:
            points = charging_points(item.get("points"))
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        for minute, energy in points:
            rows.append(dict(zip(CHARGING_COLUMNS, (
                str(item.get("id", label)), label, item.get("group", ""), minute, energy,
                positive_number(meta.get("chargerPowerW")),
                positive_number(meta.get("fullDurationMinutes")),
                positive_number(meta.get("energyWh")),
                positive_number(meta.get("averagePowerW")), meta.get("soc", ""),
                "https://www.socpk.com/charge-curves",
            ))))
    if not rows:
        raise ValueError("No charging devices selected.")
    return rows


def charging_profiles(frame):
    """Keep sources separate and derive interval-average input power as ΔWh/Δh."""
    profiles = []
    for label, rows in frame.groupby("__label", sort=False):
        series = []
        for source, samples in rows.groupby("__source", sort=False):
            points = charging_points(samples[["chargingMinutes", "chargedEnergyWh"]].values.tolist())
            power = [[(left[0] + right[0]) / 2,
                      60 * (right[1] - left[1]) / (right[0] - left[0])]
                     for left, right in zip(points, points[1:])]
            first = samples.iloc[0]
            duration = positive_number(first.get("fullDurationMinutes"))
            energy = positive_number(first.get("totalEnergyWh"))
            average = positive_number(first.get("averageChargingPowerW"))
            # Summary metadata describes the complete charge. A curve endpoint
            # alone does not establish that the device reached full charge.
            if average is None and duration is not None and energy is not None:
                average = 60 * energy / duration
            series.append({
                "source": str(source), "points": points, "powerPoints": power,
                "durationMinutes": duration, "totalEnergyWh": energy,
                "averagePowerW": average,
                "peakPowerW": max((point[1] for point in power), default=None),
            })
        first = rows.iloc[0]
        # Never silently merge different charging runs into one ranking result.
        summary = series[0] if len(series) == 1 else {}
        profiles.append({
            "label": str(label), "legend": str(label), "leader": str(label),
            "brand": str(first.get("brand", "")), "soc": str(first.get("soc", "")),
            "chargerPowerW": positive_number(first.get("chargerPowerW")),
            "series": series, "sources": [item["source"] for item in series],
            "count": sum(len(item["points"]) for item in series),
            **{key: summary.get(key) for key in
               ("durationMinutes", "totalEnergyWh", "averagePowerW", "peakPowerW")},
        })
    return sorted(profiles, key=lambda item: item["label"].casefold())

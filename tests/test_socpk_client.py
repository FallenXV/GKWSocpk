from __future__ import annotations

import base64
import json
import unittest

from Battery import battery_parser
from socpk_client import (
    RANKINGS_PAYLOAD_KEY,
    decode_embedded_payload,
    discover_module_urls,
    extract_axis_geometry,
    extract_curve_coordinates,
)


def make_bundle(payload: object, key: str) -> str:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    key_bytes = key.encode("utf-8")
    encrypted = bytes(
        byte ^ key_bytes[index % len(key_bytes)]
        for index, byte in enumerate(raw)
    )
    encoded = base64.b64encode(encrypted).decode("ascii")
    return f"var data=decode(`{encoded}`,`{key}`);"


class SocpkClientTests(unittest.TestCase):
    def test_discovers_versioned_module(self) -> None:
        html = '<script type="module" crossorigin src="/assets/index-abc.js"></script>'
        self.assertEqual(
            discover_module_urls(html, "https://www.socpk.com/batlife/"),
            ["https://www.socpk.com/assets/index-abc.js"],
        )

    def test_decodes_rankings_and_battery_rows(self) -> None:
        rows = [["苹果", "Phone", 600, "OS", 5000, 20.0, "NA"]]
        bundle = make_bundle({"battery50": rows}, RANKINGS_PAYLOAD_KEY)
        self.assertEqual(
            decode_embedded_payload(bundle, RANKINGS_PAYLOAD_KEY)["battery50"],
            rows,
        )
        self.assertEqual(battery_parser.parse_battery_rows(bundle), rows)
        future_bundle = make_bundle({"battery50": rows}, "socpk-rankings-2027")
        self.assertEqual(
            decode_embedded_payload(future_bundle, RANKINGS_PAYLOAD_KEY)["battery50"],
            rows,
        )

    def test_extracts_inset_axis_geometry(self) -> None:
        svg = """
        <svg>
          <path d="M0,648h1152V0H0v648Z"/>
          <path d="M144,576.7h892.8"/>
          <path d="M144,576.7V77.8"/>
        </svg>
        """
        self.assertEqual(
            extract_axis_geometry(svg, power_range=18, score_range=4000),
            {
                "X_START": 144.0,
                "X_WIDTH": 892.8,
                "POWER_RANGE": 18.0,
                "Y_BASE": 576.7,
                "Y_HEIGHT": 498.90000000000003,
                "SCORE_RANGE": 4000.0,
            },
        )

    def test_extracts_line_and_scatter_coordinates(self) -> None:
        line_svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="line2d_1"><path d="M 1 2 L 3.5 4.5"/></g>
        </svg>
        """
        scatter_svg = """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="PathCollection_1">
            <use href="#point" y="8" x="7"/>
            <use href="#point" x="9.5" y="10.5"/>
          </g>
        </svg>
        """
        self.assertEqual(extract_curve_coordinates(line_svg), [(1.0, 2.0), (3.5, 4.5)])
        self.assertEqual(
            extract_curve_coordinates(scatter_svg),
            [(7.0, 8.0), (9.5, 10.5)],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from battery_soc import (
    canonical_brand,
    canonical_soc_name,
    choose_phone_url,
    parse_maker_links,
    parse_phone_catalog,
)
from Battery.battery_parser import compute_soc_averages


class BatterySoCTests(unittest.TestCase):
    def test_normalizes_brands_chipsets_and_chinese_model_suffixes(self) -> None:
        self.assertEqual(canonical_brand("红米"), "xiaomi")
        self.assertEqual(canonical_brand("iQOO"), "vivo")
        self.assertEqual(
            canonical_soc_name("Qualcomm SM8750-AB Snapdragon 8 Elite (3 nm)"),
            "Snapdragon 8 Elite",
        )
        self.assertEqual(canonical_soc_name("Mediatek Dimensity 9400+ (3 nm)"),
                         "Dimensity 9400+")
        candidates = [
            ("Redmi K90 Ultra", "https://example.test/k90"),
            ("Redmi K90 Pro Max", "https://example.test/k90-pro-max"),
        ]
        self.assertEqual(choose_phone_url("红米", "K90 至尊版", candidates),
                         "https://example.test/k90")
        regional_candidates = [
            ("iQOO Neo11 (China)", "https://example.test/neo11"),
            ("iQOO Neo10 Pro+ (China)", "https://example.test/neo10-pro-plus"),
            ("GT8 (China)", "https://example.test/gt8"),
        ]
        self.assertEqual(choose_phone_url("iQOO", "Neo 11", regional_candidates),
                         "https://example.test/neo11")
        self.assertEqual(choose_phone_url("iQOO", "Neo 10 Pro+", regional_candidates),
                         "https://example.test/neo10-pro-plus")
        self.assertEqual(choose_phone_url("Realme", "GT8", regional_candidates),
                         "https://example.test/gt8")

    def test_parses_catalog_pages(self) -> None:
        makers = parse_maker_links(
            '<div class="st-text"><a href="apple-phones-48.php">Apple 152 devices</a></div>'
        )
        self.assertEqual(makers["apple"], "https://www.gsmarena.com/apple-phones-48.php")
        phones, pages = parse_phone_catalog(
            '<div class="makers"><ul><li><a href="apple_iphone_17-1.php">'
            'iPhone 17</a></li></ul></div><div class="nav-pages">'
            '<a href="apple-phones-f-48-0-p2.php">2</a></div>'
        )
        self.assertEqual(phones[0], ("iPhone 17", "https://www.gsmarena.com/apple_iphone_17-1.php"))
        self.assertEqual(pages, ["https://www.gsmarena.com/apple-phones-f-48-0-p2.php"])

    def test_embeds_per_soc_averages_in_records(self) -> None:
        records = [
            {"chipset": "Qualcomm SM8750 Snapdragon 8 Elite (3 nm)", "capacityWh": 20,
             "avgPowerW": 2, "minPerWh": 30, "hours": 10},
            {"soc": "Snapdragon 8 Elite", "capacityWh": 24,
             "avgPowerW": 3, "minPerWh": 25, "hours": 9},
            {"capacityWh": 18, "avgPowerW": 2, "minPerWh": 27, "hours": 8},
        ]
        compute_soc_averages(records)
        self.assertEqual(records[0]["soc"], "Snapdragon 8 Elite")
        self.assertEqual(records[0]["soc_device_count"], 2)
        self.assertEqual(records[1]["soc_avg_capacity_wh"], 22)
        self.assertIsNone(records[2]["soc"])


if __name__ == "__main__":
    unittest.main()

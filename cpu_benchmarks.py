"""Shared CPU benchmark schemas for exports, analysis, and the dashboard."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CpuBenchmark:
    dataset_key: str
    title: str
    slug: str
    suite: str
    score_column: str
    filename: str
    single_core: bool = False


CPU_BENCHMARKS = {
    "GB6": CpuBenchmark(
        "CPU", "Geekbench 6 multi-core", "mobile-soc-efficiency-gb7", "GB6",
        "GB6_Multi_Score", "cpu_gb6_curves.csv",
    ),
    "GB7": CpuBenchmark(
        "CPU GB7", "Geekbench 7 multi-core", "mobile-soc-efficiency-gb7", "GB7",
        "GB7_Multi_Score", "cpu_gb7_curves.csv",
    ),
    "SPEC2026_INT": CpuBenchmark(
        "SPEC INT", "SPEC CPU 2026 integer single-core", "mobile-soc-spec26", "INT",
        "SPEC2026_INT_Score", "cpu_spec2026_int_curves.csv", True,
    ),
    "SPEC2026_FP": CpuBenchmark(
        "SPEC FP", "SPEC CPU 2026 floating-point single-core", "mobile-soc-spec26", "FP",
        "SPEC2026_FP_Score", "cpu_spec2026_fp_curves.csv", True,
    ),
}

CORE_COLUMNS = ("Core", "Core_Group", "Core_Variant")


def cpu_profile_label(cpu: str, core: str = "", group: str = "", variant: str = "") -> str:
    """Identify each SPEC core independently, including unnamed core groups."""
    details = " · ".join(str(value).strip() for value in (core, group, variant) if str(value).strip())
    return f"{cpu} — {details}" if details else str(cpu)

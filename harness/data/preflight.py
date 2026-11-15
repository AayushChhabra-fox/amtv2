"""Pure checks for the dataset preflight slice.

The Spark job explains each fragment and materializes one hour of
``search_val``. These helpers decide whether that plan is safe and how many
calendar days the split covers. They do not start a cluster.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from harness.data.contract import DatasetSpec, parse_label_window, parse_span

_SCAN = re.compile(
    r"(FileScan|BatchScan)(?P<body>.*?)(?:PartitionFilters:\s*\[(?P<filters>[^\]]*)\])",
    re.DOTALL,
)


def search_val_day_count(spec: DatasetSpec) -> int:
    """Distinct calendar days in the half-open ``search_val`` window."""

    start, end = parse_span(spec.split.search_val)
    if end <= start:
        return 0
    last = end - timedelta(microseconds=1)
    day = start.date()
    count = 0
    while day <= last.date():
        count += 1
        day += timedelta(days=1)
    return count


def slice_bounds(spec: DatasetSpec, window: str) -> tuple[datetime, datetime]:
    """First ``window`` of ``search_val``, clipped to the split.

    ``window`` uses the label-window shape (``1h``, ``30m``, ``1d``).
    """

    start, end = parse_span(spec.split.search_val)
    stop = start + parse_label_window(window)
    if stop > end:
        stop = end
    if stop <= start:
        raise ValueError(f"preflight slice {window} is empty inside search_val")
    return start, stop


def explain_issues(plan: str, spec: DatasetSpec) -> list[dict]:
    """Cartesian joins, unbounded range joins, and unpartitioned scans.

    An empty plan has no issues. Partition filters are required only for
    sources that declare ``partition_filter``.
    """

    issues: list[dict] = []
    compact = re.sub(r"\s+", "", plan).lower()
    if "cartesianproduct" in compact:
        issues.append({"kind": "cartesian", "detail": "CartesianProduct"})
    if "broadcastnestedloopjoin" in compact:
        issues.append({"kind": "unbounded_range_join", "detail": "BroadcastNestedLoopJoin"})
    partitioned = {
        source.table: name
        for name, source in spec.sources.items()
        if source.partition_filter and not source.unpartitioned
    }
    for match in _SCAN.finditer(plan):
        body = match.group("body")
        filters = (match.group("filters") or "").strip()
        if filters:
            continue
        for table, name in partitioned.items():
            if table in body:
                issues.append(
                    {
                        "kind": "unpartitioned_scan",
                        "source": name,
                        "table": table,
                        "detail": "PartitionFilters: []",
                    }
                )
                break
    return issues

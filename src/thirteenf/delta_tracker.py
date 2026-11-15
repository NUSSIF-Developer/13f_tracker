"""Asset-by-quarter change matrices (pivot style) for every tracked fund."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thirteenf.filing_monitor import _prepare

METRICS = {
    "shares_pct": "Shares change (%)",
    "shares_abs": "Shares change (#)",
    "weight_pp": "Weight change (pp)",
}


def _quarter_order(values) -> list[str]:
    return sorted({str(v) for v in values}, key=lambda q: pd.Period(q, freq="Q").ordinal)


def build_delta_long(snapshot: pd.DataFrame) -> pd.DataFrame:
    """One row per manager, asset and quarter versus the manager's prior report."""
    columns = ["manager_code", "quarter", "asset", "security_id", "shares_pct", "shares_abs", "weight_pp"]
    if snapshot.empty:
        return pd.DataFrame(columns=columns)
    metadata, holdings = _prepare(snapshot)
    rows: list[pd.DataFrame] = []
    for manager_code, reports in metadata.groupby("manager_code", sort=False):
        quarters = reports.sort_values("report_period")["quarter"].tolist()
        by_quarter = {
            quarter: group.set_index("security_id")
            for quarter, group in holdings[holdings["manager_code"].eq(manager_code)].groupby("quarter")
        }
        for previous_q, current_q in zip(quarters, quarters[1:]):
            previous, current = by_quarter.get(previous_q), by_quarter.get(current_q)
            if previous is None or current is None:
                continue
            ids = current.index.union(previous.index)
            cur, prev = current.reindex(ids), previous.reindex(ids)
            shares_cur, shares_prev = cur["shares"].fillna(0.0), prev["shares"].fillna(0.0)
            shares_abs = shares_cur - shares_prev
            # A position opened from zero has no defined percentage; inf is rendered as "New".
            with np.errstate(divide="ignore", invalid="ignore"):
                shares_pct = np.where(shares_prev > 0, shares_abs / shares_prev * 100.0, np.where(shares_abs > 0, np.inf, np.nan))
            asset = cur["asset"].fillna(prev["asset"])
            rows.append(pd.DataFrame({
                "manager_code": manager_code,
                "quarter": current_q,
                "asset": asset.to_numpy(),
                "security_id": ids,
                "shares_pct": shares_pct,
                "shares_abs": shares_abs.to_numpy(),
                "weight_pp": (cur["portfolio_weight"].fillna(0.0) - prev["portfolio_weight"].fillna(0.0)).to_numpy(),
            }))
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def build_delta_matrix(
    long: pd.DataFrame,
    manager_code: str,
    metric: str = "shares_pct",
    quarters: list[str] | None = None,
) -> pd.DataFrame:
    """Assets as rows, quarters as columns; unchanged positions are blank."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    data = long[long["manager_code"].eq(manager_code)]
    columns = quarters if quarters is not None else _quarter_order(data["quarter"])
    if data.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="asset"), dtype=float)
    values = data.assign(value=data[metric].astype(float))
    values = values[values["value"].ne(0) & values["value"].notna()]
    matrix = values.pivot_table(index="asset", columns="quarter", values="value", aggfunc="sum")
    matrix = matrix.reindex(columns=columns)
    return matrix.sort_index(key=lambda index: index.str.lower())


def _fmt(metric: str):
    def render(value: float) -> str:
        if pd.isna(value):
            return ""
        if np.isinf(value):
            return "New"
        if metric == "shares_abs":
            return f"{value:+,.0f}"
        if metric == "shares_pct":
            return f"{value:+.1f}%"
        return f"{value:+.2f}"
    return render


def style_delta_matrix(matrix: pd.DataFrame, metric: str = "shares_pct"):
    """Green for increases, red for decreases, intensity scaled to magnitude."""
    finite = matrix.to_numpy(dtype=float)
    finite = np.abs(finite[np.isfinite(finite)])
    scale = float(np.percentile(finite, 95)) if finite.size else 1.0
    scale = scale or 1.0

    def colour(value: float) -> str:
        if pd.isna(value):
            return ""
        if np.isinf(value):
            return "background-color: rgba(0, 128, 0, 0.85); color: white; font-weight: 600"
        alpha = 0.15 + 0.65 * min(abs(value) / scale, 1.0)
        rgb = "46, 160, 67" if value > 0 else "218, 54, 51"
        return f"background-color: rgba({rgb}, {alpha:.2f})"

    styler = matrix.style
    apply = styler.map if hasattr(styler, "map") else styler.applymap
    return apply(colour).format(_fmt(metric), na_rep="")

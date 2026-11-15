import numpy as np
import pandas as pd

from thirteenf.delta_tracker import build_delta_long, build_delta_matrix, style_delta_matrix


def _row(manager, quarter, ticker, shares, weight):
    return {
        "manager_code": manager,
        "fund": manager,
        "report_period": quarter,
        "acceptance_datetime": f"{pd.Timestamp(quarter) + pd.Timedelta(days=40):%Y-%m-%d}",
        "cusip": f"{abs(hash(ticker)) % 1_000_000_000:09d}",
        "ticker_clean": ticker,
        "issuer": ticker,
        "put_call": "",
        "portfolio_weight": weight,
        "market_value_usd": weight * 1_000_000,
        "sector": "Tech",
        "shares": shares,
    }


def _snapshot():
    return pd.DataFrame([
        _row("AM", "2025-03-31", "A", 100, 50),
        _row("AM", "2025-03-31", "B", 100, 50),
        _row("AM", "2025-06-30", "A", 150, 60),
        _row("AM", "2025-06-30", "C", 10, 40),
        _row("AM", "2025-09-30", "A", 150, 55),
        _row("AM", "2025-09-30", "C", 5, 45),
    ])


def test_share_changes_cover_adds_new_exit_and_unchanged():
    long = build_delta_long(_snapshot())
    pct = build_delta_matrix(long, "AM", "shares_pct")

    assert pct.loc["A", "2025Q2"] == 50.0
    assert np.isinf(pct.loc["C", "2025Q2"])
    assert pct.loc["B", "2025Q2"] == -100.0
    assert pd.isna(pct.loc["A", "2025Q3"])  # unchanged shares despite weight drift
    assert pct.loc["C", "2025Q3"] == -50.0


def test_weight_metric_and_pending_quarter_column():
    long = build_delta_long(_snapshot())
    weight = build_delta_matrix(long, "AM", "weight_pp", ["2025Q2", "2025Q3", "2025Q4"])

    assert weight.loc["A", "2025Q3"] == -5.0
    assert weight["2025Q4"].isna().all()


def test_unfiled_fund_is_empty_and_styling_renders():
    long = build_delta_long(_snapshot())
    assert build_delta_matrix(long, "BRK").empty
    html = style_delta_matrix(build_delta_matrix(long, "AM")).to_html()
    assert "New" in html and "rgba(218" in html

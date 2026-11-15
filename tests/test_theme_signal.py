import pandas as pd
import pytest

from thirteenf.theme_signal import (
    build_duquesne_position_changes,
    build_duquesne_theme_allocation_history,
    summarize_theme_position_changes,
)
from thirteenf.visuals import build_theme_allocation_history_chart


def _holding(quarter, ticker, cusip, shares, weight, *, manager="DUQ", put_call="", share_type="SH"):
    return {
        "manager_code": manager,
        "fund": "Stanley Druckenmiller - Duquesne",
        "report_period": quarter,
        "acceptance_datetime": f"{pd.Timestamp(quarter) + pd.Timedelta(days=40):%Y-%m-%d}T12:00:00Z",
        "source_url": f"https://www.sec.gov/{quarter}/{ticker}",
        "cusip": cusip,
        "ticker_clean": ticker,
        "issuer": ticker,
        "share_type": share_type,
        "put_call": put_call,
        "shares": shares,
        "portfolio_weight": weight,
        "market_value_usd": weight * 100_000,
    }


def test_share_increase_is_independent_of_weight_change_and_is_theme_tagged():
    snapshot = pd.DataFrame([
        _holding("2025-03-31", "NVDA", "67066G104", 100, 10),
        _holding("2025-06-30", "NVDA", "67066G104", 150, 8),
    ])
    tags = pd.DataFrame([{
        "cusip": "67066G104", "ticker": "NVDA", "theme": "AI & Compute Infrastructure",
        "confidence": 0.94, "evidence": "GPU and data-center products",
        "source_url": "https://example.test/nvda", "model_id": "tagger-v1",
    }])

    changes = build_duquesne_position_changes(snapshot, tags)
    increase = changes.iloc[0]

    assert increase["change_type"] == "Share increase"
    assert increase["share_change"] == 50
    assert increase["weight_change_pp"] == -2
    assert increase["themes"] == "AI & Compute Infrastructure"
    assert increase["classification_confidence"] == 0.94
    assert increase["classification_evidence"] == "GPU and data-center products"
    assert increase["accepted_at"] == pd.Timestamp("2025-08-09 12:00:00+0000", tz="UTC")


def test_new_increased_decreased_unchanged_and_exited_positions_are_distinguished():
    snapshot = pd.DataFrame([
        _holding("2025-03-31", "KEEP", "000000001", 100, 10),
        _holding("2025-03-31", "CUT", "000000002", 100, 10),
        _holding("2025-03-31", "EXIT", "000000003", 100, 10),
        _holding("2025-06-30", "KEEP", "000000001", 100, 10),
        _holding("2025-06-30", "CUT", "000000002", 80, 8),
        _holding("2025-06-30", "EXIT", "000000003", 0, 0),
        _holding("2025-06-30", "NEW", "000000004", 25, 2),
        _holding("2025-06-30", "OPTION", "000000005", 500, 5, put_call="Call"),
    ])

    changes = build_duquesne_position_changes(snapshot).set_index("ticker")

    assert changes["change_type"].to_dict() == {
        "NEW": "New position",
        "KEEP": "Unchanged",
        "CUT": "Share decrease",
        "EXIT": "Full exit",
    }
    assert changes.loc["EXIT", "accepted_at"] == pd.Timestamp("2025-08-09 12:00:00+0000", tz="UTC")
    assert "OPTION" not in changes.index


def test_signal_is_duquesne_only_and_theme_summary_filters_confidence():
    snapshot = pd.DataFrame([
        _holding("2025-03-31", "BASE", "000000011", 100, 10),
        _holding("2025-06-30", "BASE", "000000011", 100, 10),
        _holding("2025-06-30", "AI1", "000000012", 10, 4),
        _holding("2025-06-30", "AI2", "000000013", 20, 6),
        _holding("2025-06-30", "PEER", "000000014", 90, 10, manager="AM"),
    ])
    tags = pd.DataFrame([
        {"ticker": "AI1", "theme": "AI & Compute Infrastructure", "confidence": 0.9},
        {"ticker": "AI2", "theme": "AI & Compute Infrastructure", "confidence": 0.6},
    ])

    changes = build_duquesne_position_changes(snapshot, tags)
    changed = changes[changes["change_type"].ne("Unchanged")]
    assert set(changed["ticker"]) == {"AI1", "AI2"}

    summary = summarize_theme_position_changes(changed, min_confidence=0.8)
    assert len(summary) == 1
    assert summary.iloc[0]["theme"] == "AI & Compute Infrastructure"
    assert summary.iloc[0]["new_positions"] == 1
    assert summary.iloc[0]["increased_positions"] == 0
    assert summary.iloc[0]["current_value_usd"] == 400_000
    assert summary.iloc[0]["classified_positions"] == 1


def test_grouped_taxonomy_rows_expand_to_tickers_and_public_profile_urls():
    snapshot = pd.DataFrame([
        _holding("2025-03-31", "NVDA", "67066G104", 100, 10),
        _holding("2025-06-30", "NVDA", "67066G104", 120, 11),
        _holding("2025-03-31", "AMD", "007903107", 50, 5),
        _holding("2025-06-30", "AMD", "007903107", 60, 6),
    ])
    taxonomy = pd.DataFrame([{
        "theme": "AI & Compute Infrastructure",
        "tickers": "NVDA; AMD",
        "confidence": 0.88,
        "evidence": "Public issuer descriptions; broad primary-business category.",
        "source_url": "https://finance.yahoo.com/quote/{ticker}/profile/",
        "model_id": "copilot-public-web-review-2026-10",
    }])

    changes = build_duquesne_position_changes(snapshot, taxonomy)
    tagged = changes.set_index("ticker")

    assert tagged.loc["NVDA", "themes"] == "AI & Compute Infrastructure"
    assert tagged.loc["AMD", "classification_source_url"] == "https://finance.yahoo.com/quote/amd/profile/"


def test_ticker_cannot_be_assigned_to_conflicting_themes():
    snapshot = pd.DataFrame([
        _holding("2025-03-31", "NVDA", "67066G104", 100, 10),
        _holding("2025-06-30", "NVDA", "67066G104", 120, 11),
    ])
    taxonomy = pd.DataFrame([
        {"ticker": "NVDA", "theme": "AI & Compute Infrastructure"},
        {"ticker": "NVDA", "theme": "Software & Cybersecurity"},
    ])

    with pytest.raises(ValueError, match="Tickers cannot be assigned to multiple themes"):
        build_duquesne_position_changes(snapshot, taxonomy)


def test_theme_history_covers_all_quarters_and_keeps_unmapped_weight_visible():
    snapshot = pd.DataFrame([
        _holding("2014-03-31", "AI1", "000000021", 100, 25),
        _holding("2014-03-31", "OTHER", "000000022", 300, 75),
        _holding("2014-03-31", "OPTION", "000000023", 1000, 50, put_call="Call"),
        _holding("2014-06-30", "AI1", "000000021", 50, 14.2857),
        _holding("2014-06-30", "OTHER", "000000022", 300, 85.7143),
    ])
    taxonomy = pd.DataFrame([{
        "ticker": "AI1", "theme": "AI & Compute Infrastructure", "confidence": 0.9,
    }])

    history = build_duquesne_theme_allocation_history(snapshot, taxonomy)
    allocation_sums = history.groupby("quarter")["portfolio_weight"].sum()

    assert set(history["quarter"]) == {"2014Q1", "2014Q2"}
    assert allocation_sums.to_dict() == pytest.approx({"2014Q1": 100.0, "2014Q2": 100.0})
    first = history[history["quarter"].eq("2014Q1")].set_index("theme")
    assert first.loc["AI & Compute Infrastructure", "portfolio_weight"] == 25
    assert first.loc["Unclassified", "portfolio_weight"] == 75
    assert "OPTION" not in set(history["theme"])
    assert first["unclassified_value_usd"].eq(7_500_000).all()

    chart = build_theme_allocation_history_chart(history)
    assert len(chart.data) == 2
    assert list(chart.layout.yaxis.range) == [0, 100]
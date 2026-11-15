from __future__ import annotations

import pandas as pd


DUQUESNE_MANAGER_CODE = "DUQ"
ALLOWED_THEMES = (
    "AI & Compute Infrastructure",
    "Digital Assets & Mining",
    "Software & Cybersecurity",
    "Healthcare & Biotechnology",
    "Financial Services & Payments",
    "Consumer & Retail",
    "Media & Communications",
    "Industrials & Transportation",
    "Energy & Power",
    "Materials & Natural Resources",
    "Real Estate & Data Centers",
    "Broad-Market & Regional Funds",
)
THEME_CLASSIFICATION_COLUMNS = [
    "cusip", "ticker", "theme", "confidence", "evidence", "source_url", "model_id",
]
POSITION_CHANGE_COLUMNS = [
    "accepted_at", "quarter", "prior_quarter", "fund", "change_type", "asset",
    "ticker", "issuer", "cusip", "share_type", "previous_shares", "current_shares",
    "share_change", "previous_weight", "current_weight", "weight_change_pp",
    "previous_value_usd", "current_value_usd", "themes", "classification_confidence",
    "classification_evidence", "classification_source_url", "classification_model_id",
    "source_url",
]
THEME_ALLOCATION_COLUMNS = [
    "quarter", "report_period", "accepted_at", "theme", "market_value_usd",
    "portfolio_weight", "position_count", "unclassified_value_usd",
]


def _clean_text(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def _theme_map(classifications: pd.DataFrame | None) -> dict[str, list[dict[str, object]]]:
    if classifications is None or classifications.empty:
        return {}
    if "theme" not in classifications:
        raise KeyError("Theme classifications require a 'theme' column")

    mapping = classifications.copy()
    if "tickers" in mapping:
        grouped_rows = mapping[_clean_text(mapping.get("ticker", pd.Series("", index=mapping.index))).eq("")].copy()
        if not grouped_rows.empty:
            grouped_rows["ticker"] = _clean_text(
                grouped_rows["tickers"],
            ).str.split(r"\s*[;,|]\s*")
            grouped_rows = grouped_rows.explode("ticker")
            source_urls = _clean_text(
                grouped_rows.get("source_url", pd.Series("", index=grouped_rows.index)),
            )
            grouped_rows["source_url"] = [
                source_url.replace("{ticker}", ticker.lower())
                for source_url, ticker in zip(source_urls, grouped_rows["ticker"].astype(str))
            ]
            mapping = pd.concat([mapping, grouped_rows], ignore_index=True)
    for column in THEME_CLASSIFICATION_COLUMNS:
        if column not in mapping:
            mapping[column] = ""
    mapping["cusip_key"] = _clean_text(mapping["cusip"]).str.upper().str.zfill(9)
    mapping["ticker_key"] = _clean_text(mapping["ticker"]).str.upper()
    mapping["theme"] = _clean_text(mapping["theme"])
    invalid_themes = sorted(set(mapping.loc[mapping["theme"].ne(""), "theme"]) - set(ALLOWED_THEMES))
    if invalid_themes:
        raise ValueError(f"Theme classifications contain labels outside the fixed taxonomy: {invalid_themes}")
    mapping["confidence"] = pd.to_numeric(mapping["confidence"], errors="coerce")
    ticker_theme_counts = mapping[mapping["ticker_key"].ne("")].groupby("ticker_key")["theme"].nunique()
    conflicting_tickers = sorted(ticker_theme_counts[ticker_theme_counts.gt(1)].index)
    if conflicting_tickers:
        raise ValueError(f"Tickers cannot be assigned to multiple themes: {conflicting_tickers}")
    result: dict[str, list[dict[str, object]]] = {}
    for row in mapping[mapping["theme"].ne("")].to_dict("records"):
        keys = []
        if row["cusip_key"] and row["cusip_key"] != "000000000":
            keys.append(f"CUSIP:{row['cusip_key']}")
        if row["ticker_key"]:
            keys.append(f"TICKER:{row['ticker_key']}")
        for key in keys:
            result.setdefault(key, []).append(row)
    return result


def build_duquesne_position_changes(
    snapshot: pd.DataFrame,
    theme_classifications: pd.DataFrame | None = None,
    *,
    quarters: int = 8,
    manager_code: str = DUQUESNE_MANAGER_CODE,
) -> pd.DataFrame:
    """Compare Duquesne's reported share quantities across available filings.

    The result includes all share-change states. A unit increase is a disclosed
    quantity change, not proof of a transaction or its timing.
    """
    required = {"manager_code", "report_period", "cusip", "shares"}
    missing = required - set(snapshot.columns)
    if missing:
        raise KeyError(f"Theme signal missing columns: {sorted(missing)}")
    if quarters < 1:
        raise ValueError("quarters must be at least 1")

    data = snapshot[snapshot["manager_code"].astype(str).eq(manager_code)].copy()
    if data.empty:
        return pd.DataFrame(columns=POSITION_CHANGE_COLUMNS)

    data["report_period"] = pd.to_datetime(data["report_period"], errors="coerce")
    data = data[data["report_period"].notna()].copy()
    data["quarter"] = pd.PeriodIndex(data["report_period"], freq="Q").astype(str)
    data["cusip_key"] = _clean_text(data["cusip"]).str.upper().str.zfill(9)
    data["ticker"] = _clean_text(data.get("ticker_clean", data.get("ticker", pd.Series("", index=data.index))))
    data["issuer"] = _clean_text(data.get("issuer", pd.Series("", index=data.index)))
    data["fund"] = _clean_text(data.get("fund", pd.Series("Duquesne", index=data.index)))
    data["source_url"] = _clean_text(data.get("source_url", pd.Series("", index=data.index)))
    data["share_type"] = _clean_text(data.get("share_type", pd.Series("SH", index=data.index))).str.upper()
    data["put_call"] = _clean_text(data.get("put_call", pd.Series("", index=data.index)))
    # Principal amounts and options are not common-share accumulation signals.
    data = data[data["share_type"].isin(["", "SH"]) & data["put_call"].eq("")]
    data = data[data["cusip_key"].ne("") & data["cusip_key"].ne("000000000")].copy()
    data["shares"] = pd.to_numeric(data["shares"], errors="coerce").fillna(0.0)
    data["portfolio_weight"] = pd.to_numeric(
        data.get("portfolio_weight", pd.Series(0.0, index=data.index)), errors="coerce",
    ).fillna(0.0)
    data["market_value_usd"] = pd.to_numeric(
        data.get("market_value_usd", pd.Series(0.0, index=data.index)), errors="coerce",
    ).fillna(0.0)

    accepted = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    for column in ("acceptance_datetime", "available_at", "filing_date"):
        if column in data:
            accepted = accepted.fillna(pd.to_datetime(data[column], errors="coerce", utc=True))
    data["accepted_at"] = accepted

    grouped = (
        data.groupby(["quarter", "cusip_key"], as_index=False, dropna=False)
        .agg(
            report_period=("report_period", "max"), fund=("fund", "last"),
            ticker=("ticker", "last"), issuer=("issuer", "last"),
            share_type=("share_type", "last"), shares=("shares", "sum"),
            portfolio_weight=("portfolio_weight", "sum"),
            market_value_usd=("market_value_usd", "sum"),
            accepted_at=("accepted_at", "max"), source_url=("source_url", "last"),
        )
        .sort_values(["report_period", "accepted_at"])
        .reset_index(drop=True)
    )
    quarter_metadata = (
        grouped.groupby("quarter", as_index=False)
        .agg(accepted_at=("accepted_at", "max"), source_url=("source_url", "last"))
        .set_index("quarter")
    )
    available_quarters = sorted(
        grouped["quarter"].unique(), key=lambda value: pd.Period(value, freq="Q").ordinal,
    )[-quarters:]
    grouped = grouped[grouped["quarter"].isin(available_quarters)]
    classification_map = _theme_map(theme_classifications)
    records: list[dict[str, object]] = []

    for quarter_index in range(1, len(available_quarters)):
        current_quarter = available_quarters[quarter_index]
        prior_quarter = available_quarters[quarter_index - 1]
        current = grouped[grouped["quarter"].eq(current_quarter)].set_index("cusip_key")
        previous = grouped[grouped["quarter"].eq(prior_quarter)].set_index("cusip_key")
        for cusip in current.index.union(previous.index):
            current_row = current.loc[cusip] if cusip in current.index else None
            previous_row = previous.loc[cusip] if cusip in previous.index else None
            current_shares = float(current_row["shares"]) if current_row is not None else 0.0
            previous_shares = float(previous_row["shares"]) if previous_row is not None else 0.0
            if previous_shares <= 0 < current_shares:
                change_type = "New position"
            elif previous_shares > 0 >= current_shares:
                change_type = "Full exit"
            elif current_shares > previous_shares:
                change_type = "Share increase"
            elif current_shares < previous_shares:
                change_type = "Share decrease"
            else:
                change_type = "Unchanged"

            row = current_row if current_row is not None else previous_row
            assert row is not None
            tags = classification_map.get(f"CUSIP:{cusip}", [])
            if not tags and row["ticker"]:
                tags = classification_map.get(f"TICKER:{str(row['ticker']).upper()}", [])
            unique_tags = {str(tag["theme"]): tag for tag in tags}
            confidence_values = [
                float(tag["confidence"]) for tag in unique_tags.values()
                if pd.notna(tag["confidence"])
            ]
            evidence = sorted({str(tag["evidence"]) for tag in unique_tags.values() if tag["evidence"]})
            source_urls = sorted({str(tag["source_url"]) for tag in unique_tags.values() if tag["source_url"]})
            model_ids = sorted({str(tag["model_id"]) for tag in unique_tags.values() if tag["model_id"]})
            current_weight = float(current_row["portfolio_weight"]) if current_row is not None else 0.0
            previous_weight = float(previous_row["portfolio_weight"]) if previous_row is not None else 0.0
            records.append({
                "accepted_at": quarter_metadata.at[current_quarter, "accepted_at"], "quarter": current_quarter,
                "prior_quarter": prior_quarter, "fund": row["fund"],
                "change_type": change_type,
                "asset": row["ticker"] or row["issuer"] or cusip,
                "ticker": row["ticker"], "issuer": row["issuer"], "cusip": cusip,
                "share_type": row["share_type"], "previous_shares": previous_shares,
                "current_shares": current_shares, "share_change": current_shares - previous_shares,
                "previous_weight": previous_weight, "current_weight": current_weight,
                "weight_change_pp": current_weight - previous_weight,
                "previous_value_usd": float(previous_row["market_value_usd"]) if previous_row is not None else 0.0,
                "current_value_usd": float(current_row["market_value_usd"]) if current_row is not None else 0.0,
                "themes": "; ".join(sorted(unique_tags)),
                "classification_confidence": min(confidence_values) if confidence_values else pd.NA,
                "classification_evidence": " | ".join(evidence),
                "classification_source_url": " | ".join(source_urls),
                "classification_model_id": " | ".join(model_ids),
                "source_url": quarter_metadata.at[current_quarter, "source_url"],
            })

    if not records:
        return pd.DataFrame(columns=POSITION_CHANGE_COLUMNS)
    result = pd.DataFrame(records)
    result["_change_order"] = result["change_type"].map({"New position": 0, "Share increase": 1})
    return (
        result.sort_values(["accepted_at", "_change_order", "share_change"], ascending=[False, True, False])
        .drop(columns="_change_order")
        .reset_index(drop=True)[POSITION_CHANGE_COLUMNS]
    )


def summarize_theme_position_changes(
    position_changes: pd.DataFrame,
    *,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """Aggregate new and increased disclosed shares by classified theme."""
    columns = [
        "quarter", "prior_quarter", "theme", "new_positions", "increased_positions",
        "current_value_usd", "current_weight_of_changed_positions", "weight_change_pp",
        "classified_positions",
    ]
    if position_changes.empty:
        return pd.DataFrame(columns=columns)
    required = {"change_type", "themes", "classification_confidence", "quarter", "prior_quarter"}
    missing = required - set(position_changes.columns)
    if missing:
        raise KeyError(f"Theme summary missing columns: {sorted(missing)}")
    candidates = position_changes[
        position_changes["change_type"].isin(["New position", "Share increase"])
    ].copy()
    confidence = pd.to_numeric(candidates["classification_confidence"], errors="coerce")
    candidates = candidates[confidence.ge(float(min_confidence)) & candidates["themes"].fillna("").ne("")]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    candidates["theme"] = candidates["themes"].astype(str).str.split("; ")
    candidates = candidates.explode("theme")
    summary = (
        candidates.groupby(["quarter", "prior_quarter", "theme"], as_index=False)
        .agg(
            new_positions=("change_type", lambda values: int(values.eq("New position").sum())),
            increased_positions=("change_type", lambda values: int(values.eq("Share increase").sum())),
            current_value_usd=("current_value_usd", "sum"),
            current_weight_of_changed_positions=("current_weight", "sum"),
            weight_change_pp=("weight_change_pp", "sum"),
            classified_positions=("cusip", "nunique"),
        )
        .sort_values(["quarter", "current_weight_of_changed_positions"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary[columns]


def build_duquesne_theme_allocation_history(
    snapshot: pd.DataFrame,
    theme_classifications: pd.DataFrame | None = None,
    *,
    manager_code: str = DUQUESNE_MANAGER_CODE,
) -> pd.DataFrame:
    """Build quarter-end theme weights for the manager's full available history.

    Weights are normalized to reported common-share market value for that
    quarter. Unmapped issuers remain visible as ``Unclassified``.
    """
    required = {"manager_code", "report_period", "cusip", "market_value_usd"}
    missing = required - set(snapshot.columns)
    if missing:
        raise KeyError(f"Theme allocation missing columns: {sorted(missing)}")
    data = snapshot[snapshot["manager_code"].astype(str).eq(manager_code)].copy()
    if data.empty:
        return pd.DataFrame(columns=THEME_ALLOCATION_COLUMNS)

    data["report_period"] = pd.to_datetime(data["report_period"], errors="coerce")
    data = data[data["report_period"].notna()].copy()
    data["quarter"] = pd.PeriodIndex(data["report_period"], freq="Q").astype(str)
    data["cusip_key"] = _clean_text(data["cusip"]).str.upper().str.zfill(9)
    data["ticker"] = _clean_text(data.get("ticker_clean", data.get("ticker", pd.Series("", index=data.index))))
    data["share_type"] = _clean_text(data.get("share_type", pd.Series("SH", index=data.index))).str.upper()
    data["put_call"] = _clean_text(data.get("put_call", pd.Series("", index=data.index)))
    data = data[
        data["share_type"].isin(["", "SH"])
        & data["put_call"].eq("")
        & data["cusip_key"].ne("")
        & data["cusip_key"].ne("000000000")
    ].copy()
    data["market_value_usd"] = pd.to_numeric(data["market_value_usd"], errors="coerce").fillna(0.0)

    accepted = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    for column in ("acceptance_datetime", "available_at", "filing_date"):
        if column in data:
            accepted = accepted.fillna(pd.to_datetime(data[column], errors="coerce", utc=True))
    data["accepted_at"] = accepted

    holdings = (
        data.groupby(["quarter", "cusip_key"], as_index=False, dropna=False)
        .agg(
            report_period=("report_period", "max"), ticker=("ticker", "last"),
            market_value_usd=("market_value_usd", "sum"),
            accepted_at=("accepted_at", "max"),
        )
    )
    classification_map = _theme_map(theme_classifications)
    themes = []
    for row in holdings.to_dict("records"):
        tags = classification_map.get(f"CUSIP:{row['cusip_key']}", [])
        if not tags and row["ticker"]:
            tags = classification_map.get(f"TICKER:{str(row['ticker']).upper()}", [])
        theme = sorted({str(tag["theme"]) for tag in tags})
        if len(theme) > 1:
            raise ValueError(f"Security {row['cusip_key']} maps to multiple themes: {theme}")
        themes.append(theme[0] if theme else "Unclassified")
    holdings["theme"] = themes
    holdings = holdings[holdings["market_value_usd"].gt(0)].copy()
    if holdings.empty:
        return pd.DataFrame(columns=THEME_ALLOCATION_COLUMNS)

    total_value = holdings.groupby("quarter")["market_value_usd"].transform("sum")
    holdings["portfolio_weight"] = holdings["market_value_usd"].div(total_value).mul(100)
    grouped = (
        holdings.groupby(["quarter", "theme"], as_index=False)
        .agg(
            report_period=("report_period", "max"),
            accepted_at=("accepted_at", "max"),
            market_value_usd=("market_value_usd", "sum"),
            portfolio_weight=("portfolio_weight", "sum"),
            position_count=("cusip_key", "nunique"),
        )
    )
    unmapped = (
        grouped[grouped["theme"].eq("Unclassified")]
        .groupby("quarter")["market_value_usd"].sum()
        .rename("unclassified_value_usd")
    )
    grouped = grouped.join(unmapped, on="quarter")
    grouped["unclassified_value_usd"] = grouped["unclassified_value_usd"].fillna(0.0)
    return (
        grouped.sort_values(
            ["report_period", "portfolio_weight"], ascending=[True, False],
        ).reset_index(drop=True)[THEME_ALLOCATION_COLUMNS]
    )
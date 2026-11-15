from __future__ import annotations

import html
import time
from typing import Any

import pandas as pd
import requests

from thirteenf.sec_live import filing_html_url

TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"


def top_moves(changes: pd.DataFrame, count: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the largest percentage-point increases and decreases."""
    if changes.empty:
        return changes.head(0), changes.head(0)
    increases = changes[changes["change_pp"].gt(0)].nlargest(count, "change_pp")
    decreases = changes[changes["change_pp"].lt(0)].nsmallest(count, "change_pp")
    return increases, decreases


def _lines(moves: pd.DataFrame, label_column: str, with_type: bool = False) -> list[str]:
    if moves.empty:
        return ["  none"]
    lines = []
    for row in moves.itertuples(index=False):
        values = row._asdict()
        label = html.escape(str(values[label_column]))
        tag = ""
        if with_type and values.get("change_type") in {"New position", "Full exit"}:
            tag = f" [{html.escape(values['change_type'])}]"
        lines.append(
            f"  {label}{tag}: {values['previous_weight']:.2f}% → {values['current_weight']:.2f}% "
            f"({values['change_pp']:+.2f} pp)"
        )
    return lines


def format_filing_message(
    filing: dict[str, Any], positions: pd.DataFrame, sectors: pd.DataFrame, count: int = 3,
) -> str:
    """One Telegram HTML message for a single manager filing."""
    amended = str(filing.get("form", "")).endswith("/A")
    period = pd.Timestamp(filing["report_period"]).strftime("%d %b %Y")
    accepted = pd.to_datetime(filing.get("accepted_at"), errors="coerce", utc=True)
    accepted_text = accepted.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(accepted) else "unknown"
    title = f"<b>{html.escape(str(filing['fund']))}</b>" + (" (AMENDED)" if amended else "")
    parts = [title, f"13F report period {period} · accepted {accepted_text}"]
    if positions.empty and sectors.empty:
        parts.append("No prior report available to compare.")
    else:
        pos_up, pos_down = top_moves(positions, count)
        sec_up, sec_down = top_moves(sectors, count)
        parts += [
            "", "<b>Top position increases</b>", *_lines(pos_up, "asset", True),
            "<b>Top position decreases</b>", *_lines(pos_down, "asset", True),
            "", "<b>Top sector increases</b>", *_lines(sec_up, "sector"),
            "<b>Top sector decreases</b>", *_lines(sec_down, "sector"),
        ]
    url = filing_html_url(str(filing.get("source_url", ""))) if filing.get("source_url") else ""
    if url:
        parts += ["", f'<a href="{html.escape(url)}">View filing on EDGAR</a>']
    return "\n".join(parts)


def send_telegram(
    token: str, chat_id: str, text: str, retries: int = 3, session: Any = requests,
) -> None:
    """Send one message; raise RuntimeError if every attempt fails."""
    last_error = ""
    for attempt in range(retries):
        try:
            response = session.post(
                TELEGRAM_URL.format(token=token),
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                      "disable_web_page_preview": True},
                timeout=20,
            )
            if response.ok and response.json().get("ok"):
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = type(exc).__name__
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Telegram send failed after {retries} attempts: {last_error}")

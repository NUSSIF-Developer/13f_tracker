from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from thirteenf.config import PROCESSED_DATA_DIR, SEC_USER_AGENT_ENV
from thirteenf.filing_calendar import should_poll
from thirteenf.filing_monitor import build_position_change_alerts, build_sector_change_alerts
from thirteenf.notify import format_filing_message, send_telegram
from thirteenf.runner import run_sec_ingestion

NOTIFIED_PATH = PROCESSED_DATA_DIR / "notified_accessions.json"
RECENT_DAYS = 7


def load_notified(path: Path = NOTIFIED_PATH) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()


def save_notified(accessions: set[str], path: Path = NOTIFIED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(accessions), indent=2), encoding="utf-8")


def pending_filings(history: pd.DataFrame, notified: set[str], now: pd.Timestamp) -> list[dict]:
    """Recently accepted filings that have not been announced yet, oldest first."""
    if history.empty or "accession_number" not in history:
        return []
    frame = history.copy()
    accepted = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for column in ("acceptance_datetime", "available_at", "filing_date"):
        if column in frame:
            accepted = accepted.fillna(pd.to_datetime(frame[column], errors="coerce", utc=True))
    frame["accepted_at"] = accepted
    filings = (
        frame.groupby("accession_number", as_index=False)
        .agg(
            manager_code=("manager_code", "first"), fund=("fund", "first"),
            form=("form", "first"), report_period=("report_period", "first"),
            accepted_at=("accepted_at", "max"),
            source_url=("source_url", "first") if "source_url" in frame else ("fund", "first"),
        )
    )
    cutoff = now - pd.Timedelta(days=RECENT_DAYS)
    filings = filings[
        ~filings["accession_number"].astype(str).isin(notified) & filings["accepted_at"].ge(cutoff)
    ]
    return filings.sort_values("accepted_at").to_dict("records")


def filing_alert_message(history: pd.DataFrame, filing: dict) -> str:
    manager = history[history["manager_code"].eq(filing["manager_code"])]
    quarter = str(pd.Period(pd.Timestamp(filing["report_period"]), freq="Q"))
    codes = [filing["manager_code"]]
    positions = build_position_change_alerts(manager, threshold_pp=0.0, quarters=10_000, manager_codes=codes)
    sectors = build_sector_change_alerts(manager, threshold_pp=0.0, quarters=10_000, manager_codes=codes)
    positions = positions[positions["quarter"].eq(quarter)]
    sectors = sectors[sectors["quarter"].eq(quarter)]
    return format_filing_message(filing, positions, sectors)


def notify_pending(
    history: pd.DataFrame,
    notified: set[str],
    sender: Callable[[str], None],
    now: pd.Timestamp | None = None,
    save: Callable[[set[str]], None] = save_notified,
) -> int:
    """Send one message per pending filing; record each only after it is delivered."""
    failures = 0
    for filing in pending_filings(history, notified, now or pd.Timestamp.now(tz="UTC")):
        try:
            sender(filing_alert_message(history, filing))
        except Exception as exc:
            print(f"{filing['manager_code']} {filing['accession_number']}: {exc}", file=sys.stderr)
            failures += 1
            continue
        notified.add(str(filing["accession_number"]))
        save(notified)
    return failures


def main() -> int:
    if not should_poll(datetime.now(timezone.utc)) and not os.getenv("FORCE_REFRESH"):
        print("Outside the filing window; nothing to do.")
        return 0
    failed_managers: list[str] = []

    def report_manager_error(code: str, exc: Exception) -> None:
        failed_managers.append(code)
        print(f"{code}: {type(exc).__name__}: {exc}. Retaining saved data and continuing.", file=sys.stderr)

    history = run_sec_ingestion(
        user_agent=os.getenv(SEC_USER_AGENT_ENV, ""), quarters=2, incremental=True,
        on_manager_error=report_manager_error,
    )
    if failed_managers:
        print(
            f"Incomplete SEC refresh for {', '.join(failed_managers)}; these managers will be retried "
            "on the next run. Successful updates have been saved.",
            file=sys.stderr,
        )
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("Telegram is not configured; skipping alerts.")
        return 1 if failed_managers else 0
    failures = notify_pending(history, load_notified(), lambda text: send_telegram(token, chat_id, text))
    return 1 if failed_managers or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

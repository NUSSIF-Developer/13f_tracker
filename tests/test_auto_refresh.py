from datetime import date, datetime, timezone

import pandas as pd

from thirteenf.auto_refresh import notify_pending, pending_filings
from thirteenf.filing_calendar import filing_deadline, in_polling_window, should_poll
from thirteenf.notify import format_filing_message, send_telegram


def test_deadline_rolls_weekend_to_monday():
    assert filing_deadline(date(2026, 9, 30)) == date(2026, 11, 16)
    assert filing_deadline(date(2026, 3, 31)) == date(2026, 5, 15)


def test_polling_window_covers_three_days_before_through_day_after():
    assert in_polling_window(date(2026, 11, 13))
    assert in_polling_window(date(2026, 11, 17))
    assert not in_polling_window(date(2026, 11, 12))
    assert not in_polling_window(date(2026, 11, 18))


def test_should_poll_daily_check_outside_window():
    assert should_poll(datetime(2026, 12, 10, 6, 2, tzinfo=timezone.utc))
    assert not should_poll(datetime(2026, 12, 10, 9, 2, tzinfo=timezone.utc))


def _history():
    rows = []
    for period, accession, weights in (
        ("2026-06-30", "A1", {"AAA": 50.0, "BBB": 30.0, "CCC": 20.0, "DDD": 0.0}),
        ("2026-09-30", "A2", {"AAA": 40.0, "BBB": 45.0, "CCC": 0.0, "DDD": 15.0}),
    ):
        for ticker, weight in weights.items():
            if weight:
                rows.append({
                    "manager_code": "AC", "fund": "Akre", "form": "13F-HR",
                    "report_period": period, "accession_number": accession,
                    "acceptance_datetime": "2026-11-13 21:15:00+00:00" if accession == "A2" else "2026-08-14 20:00:00+00:00",
                    "cusip": ticker * 3, "ticker": ticker, "issuer": ticker,
                    "portfolio_weight": weight, "market_value_usd": weight * 1e6,
                    "sector": "Tech" if ticker in {"AAA", "DDD"} else "Fin",
                    "source_url": "https://www.sec.gov/Archives/edgar/data/1/0001/info.xml",
                })
    return pd.DataFrame(rows)


def test_pending_filings_skips_notified_and_old():
    now = pd.Timestamp("2026-11-14", tz="UTC")
    assert [f["accession_number"] for f in pending_filings(_history(), set(), now)] == ["A2"]
    assert pending_filings(_history(), {"A2"}, now) == []


def test_notify_pending_sends_top_moves_and_records_only_on_success():
    sent, saved = [], []
    now = pd.Timestamp("2026-11-14", tz="UTC")
    notified: set[str] = set()
    failures = notify_pending(_history(), notified, sent.append, now, saved.append)
    assert failures == 0 and notified == {"A2"}
    message = sent[0]
    assert "Akre" in message and "BBB" in message and "+15.00 pp" in message
    assert "CCC [Full exit]" in message and "DDD [New position]" in message
    assert "Tech" in message and "Fin" in message

    def boom(_text):
        raise RuntimeError("down")

    pending: set[str] = set()
    assert notify_pending(_history(), pending, boom, now, saved.append) == 1
    assert pending == set()


def test_format_message_without_prior_report():
    filing = {"fund": "X & Co", "form": "13F-HR/A", "report_period": "2026-09-30",
              "accepted_at": pd.Timestamp("2026-11-13", tz="UTC"), "source_url": ""}
    text = format_filing_message(filing, pd.DataFrame(), pd.DataFrame())
    assert "X &amp; Co" in text and "AMENDED" in text and "No prior report" in text


def test_send_telegram_retries_then_raises(monkeypatch):
    class Response:
        ok = False
        status_code = 500

        def json(self):
            return {}

    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            Session.calls += 1
            return Response()

    import thirteenf.notify as notify
    monkeypatch.setattr(notify.time, "sleep", lambda _s: None)
    try:
        send_telegram("t", "c", "hi", session=Session())
    except RuntimeError as exc:
        assert "3 attempts" in str(exc)
    else:
        raise AssertionError("expected failure")
    assert Session.calls == 3

import pandas as pd
import pytest
import requests

import thirteenf.runner as runner
from thirteenf.scrapers.sec_edgar import parse_13f_xml_content
from thirteenf.sec_pipeline import attach_filing_metadata, build_as_of_snapshots, reconcile_amendments
from thirteenf.storage import load_snapshot, save_snapshot


def _raw_holding(accession: str, period: str, accepted: str, cusip: str, manager_code: str = "DUQ") -> pd.DataFrame:
    xml = f"""<informationTable><infoTable><nameOfIssuer>X</nameOfIssuer><cusip>{cusip}</cusip>
    <value>100</value><shrsOrPrnAmt><sshPrnamt>10</sshPrnamt></shrsOrPrnAmt></infoTable></informationTable>"""
    filing = {
        "cik": runner.FUND_CONFIG[manager_code]["cik"],
        "accession_number": accession,
        "form": "13F-HR",
        "report_period": period,
        "filing_date": accepted[:10],
        "acceptance_datetime": accepted,
        "content_hash": accession,
    }
    return attach_filing_metadata(
        parse_13f_xml_content(xml), filing, manager_code, f"https://www.sec.gov/{accession}"
    )


def test_incremental_refresh_preserves_history_and_replaces_only_changed_periods(tmp_path, monkeypatch):
    old = _raw_holding("OLD", "2025-12-31", "2026-02-14T20:00:00Z", "111111111")
    save_snapshot(reconcile_amendments(old), tmp_path / "sec_holdings_history.csv")
    save_snapshot(build_as_of_snapshots(old), tmp_path / "sec_holdings_events.csv")
    new = _raw_holding("NEW", "2026-03-31", "2026-05-15T20:00:00Z", "222222222")
    observed_known = {}

    monkeypatch.setattr(runner, "SECClient", lambda *_args, **_kwargs: object())

    def fake_fetch(code, _client, **kwargs):
        observed_known[code] = kwargs["known_accessions"]
        return new.copy() if code == "DUQ" else pd.DataFrame()

    monkeypatch.setattr(runner, "fetch_manager_holdings", fake_fetch)
    result = runner.run_sec_ingestion(
        output_dir=tmp_path,
        quarters=2,
        user_agent="Example research@example.com",
        incremental=True,
    )

    assert set(result["report_period"]) == {"2025-12-31", "2026-03-31"}
    assert "OLD" in observed_known["DUQ"]
    saved_events = load_snapshot(tmp_path / "sec_holdings_events.csv")
    assert set(saved_events["snapshot_accession"]) == {"OLD", "NEW"}


def test_incremental_refresh_with_no_new_filings_returns_existing_history(tmp_path, monkeypatch):
    old = _raw_holding("OLD", "2025-12-31", "2026-02-14T20:00:00Z", "111111111")
    expected = reconcile_amendments(old)
    save_snapshot(expected, tmp_path / "sec_holdings_history.csv")
    save_snapshot(build_as_of_snapshots(old), tmp_path / "sec_holdings_events.csv")

    monkeypatch.setattr(runner, "SECClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "fetch_manager_holdings", lambda *_args, **_kwargs: pd.DataFrame())

    result = runner.run_sec_ingestion(
        output_dir=tmp_path,
        user_agent="Example research@example.com",
        incremental=True,
    )
    assert list(result["accession_number"]) == ["OLD"]


@pytest.mark.parametrize("error_type", [requests.HTTPError, requests.ReadTimeout, ValueError])
def test_partial_refresh_preserves_failed_manager_and_retries_it_next_run(tmp_path, monkeypatch, error_type):
    codes = ["DUQ", "TCI", "THIEL"]
    monkeypatch.setattr(runner, "FUND_CONFIG", {code: runner.FUND_CONFIG[code] for code in codes})
    monkeypatch.setattr(runner, "SECClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    old = pd.concat([
        _raw_holding(f"{code}_OLD", "2026-06-30", "2026-08-14T20:00:00Z", "111111111", code)
        for code in codes
    ], ignore_index=True)
    history_path = tmp_path / "sec_holdings_history.csv"
    events_path = tmp_path / "sec_holdings_events.csv"
    save_snapshot(reconcile_amendments(old), history_path)
    save_snapshot(build_as_of_snapshots(old), events_path)
    error = error_type("SEC request failed")
    calls, failures = [], {}

    def fetch(code, client, **kwargs):
        calls.append(code)
        if code == "TCI":
            raise error
        return _raw_holding(f"{code}_NEW", "2026-09-30", "2026-11-16T14:00:00Z", "222222222", code)

    monkeypatch.setattr(runner, "fetch_manager_holdings", fetch)
    result = runner.run_sec_ingestion(
        output_dir=tmp_path, incremental=True, on_manager_error=lambda code, exc: failures.update({code: exc}),
    )

    expected = {"DUQ_OLD", "TCI_OLD", "THIEL_OLD", "DUQ_NEW", "THIEL_NEW"}
    assert calls == codes  # A failure in the middle does not block later managers.
    assert failures == {"TCI": error}
    assert set(result.accession_number) == expected
    assert set(load_snapshot(history_path).accession_number) == expected
    assert set(load_snapshot(events_path).snapshot_accession) == expected
    assert set(result.loc[result.manager_code.eq("TCI"), "report_period"]) == {"2026-06-30"}

    def recovered_fetch(code, client, **kwargs):
        if code == "TCI":
            assert kwargs["known_accessions"] == {"TCI_OLD"}
            return _raw_holding("TCI_NEW", "2026-09-30", "2026-11-16T14:00:00Z", "333333333", code)
        assert f"{code}_NEW" in kwargs["known_accessions"]
        return pd.DataFrame()

    monkeypatch.setattr(runner, "fetch_manager_holdings", recovered_fetch)
    failures.clear()
    result = runner.run_sec_ingestion(
        output_dir=tmp_path, incremental=True, on_manager_error=lambda code, exc: failures.update({code: exc}),
    )
    assert failures == {}
    assert set(result.accession_number) == expected | {"TCI_NEW"}
    assert set(load_snapshot(events_path).snapshot_accession) == expected | {"TCI_NEW"}


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_failed_refresh_leaves_saved_files_intact(tmp_path, monkeypatch, continue_on_error):
    monkeypatch.setattr(runner, "FUND_CONFIG", {"DUQ": runner.FUND_CONFIG["DUQ"]})
    monkeypatch.setattr(runner, "SECClient", lambda *_args, **_kwargs: object())
    old = _raw_holding("OLD", "2026-06-30", "2026-08-14T20:00:00Z", "111111111")
    history_path = save_snapshot(reconcile_amendments(old), tmp_path / "sec_holdings_history.csv")
    events_path = save_snapshot(build_as_of_snapshots(old), tmp_path / "sec_holdings_events.csv")
    before = {path: path.read_bytes() for path in (history_path, events_path)}

    def fail(*args, **kwargs):
        raise requests.ReadTimeout("SEC unavailable")

    monkeypatch.setattr(runner, "fetch_manager_holdings", fail)
    failures = []
    if continue_on_error:
        result = runner.run_sec_ingestion(
            output_dir=tmp_path, incremental=True, on_manager_error=lambda code, exc: failures.append(code),
        )
        assert list(result.accession_number) == ["OLD"]
        assert failures == ["DUQ"]
    else:
        with pytest.raises(RuntimeError, match="DUQ: ReadTimeout"):
            runner.run_sec_ingestion(output_dir=tmp_path, incremental=True)
    assert {path: path.read_bytes() for path in before} == before


def test_partial_refresh_requires_incremental_mode(tmp_path):
    with pytest.raises(ValueError, match="requires incremental=True"):
        runner.run_sec_ingestion(output_dir=tmp_path, on_manager_error=lambda *args: None)

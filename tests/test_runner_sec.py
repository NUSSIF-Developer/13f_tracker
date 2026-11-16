from unittest.mock import Mock

import pandas as pd
import pytest

import thirteenf.runner as runner
from thirteenf.runner import fetch_manager_holdings
from thirteenf.sec_live import parse_submissions


class FakeSECClient:
    def get_json(self, url):
        return {
            "cik": "1536411",
            "name": "Duquesne Family Office LLC",
            "filings": {"recent": {
                "accessionNumber": ["0001536411-26-000006"],
                "form": ["13F-HR"],
                "filingDate": ["2026-08-14"],
                "reportDate": ["2026-06-30"],
                "acceptanceDateTime": ["2026-08-14T19:55:57.000Z"],
                "primaryDocument": ["primary_doc.xml"],
            }},
        }

    def get_text(self, url):
        return """<table class="tableFile">
        <tr><td>1</td><td></td><td><a href="primary_doc.xml">primary</a></td><td>13F-HR</td><td>1</td></tr>
        <tr><td>2</td><td></td><td><a href="info.xml">info</a></td><td>INFORMATION TABLE</td><td>1</td></tr>
        </table>"""

    def get_bytes(self, url):
        if url.endswith("primary_doc.xml"):
            return b"<edgarSubmission><isAmendment>false</isAmendment><reportCalendarOrQuarter>06-30-2026</reportCalendarOrQuarter></edgarSubmission>"
        return b"""<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
        <infoTable><nameOfIssuer>Apple Inc</nameOfIssuer><cusip>037833100</cusip><value>1000000</value>
        <shrsOrPrnAmt><sshPrnamt>100</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>
        </informationTable>"""


def test_manager_ingestion_is_end_to_end_and_preserves_provenance(tmp_path):
    result = fetch_manager_holdings("DUQ", FakeSECClient(), quarters=1, raw_dir=tmp_path)
    assert len(result) == 1
    assert result.iloc[0]["cik"] == "0001536411"
    assert result.iloc[0]["cusip"] == "037833100"
    assert result.iloc[0]["portfolio_weight"] == 100.0
    assert len(result.iloc[0]["content_hash"]) == 64
    assert (tmp_path / "DUQ" / "0001536411-26-000006" / "information_table.xml").exists()


def _filings_with_legacy_entry():
    filings = parse_submissions(FakeSECClient().get_json(""))
    legacy = filings.iloc[0].to_dict()
    legacy.update(
        accession_number="0000902664-07-002098",
        report_period="2007-06-30",
        filing_date="2007-08-14",
        acceptance_datetime="2007-08-14T20:00:00.000Z",
        index_url="https://www.sec.gov/legacy-2007-index.html",
    )
    return pd.concat([pd.DataFrame([legacy]), filings], ignore_index=True)


def test_incremental_refresh_does_not_download_unsupported_old_filings(tmp_path, monkeypatch):
    filings = _filings_with_legacy_entry()
    monkeypatch.setattr(runner, "fetch_sec_company_filings", lambda *args, **kwargs: filings.copy())
    client = Mock(spec=FakeSECClient)
    client.get_text.side_effect = AssertionError("A known or unsupported filing must not be downloaded")

    for _ in range(2):
        result = fetch_manager_holdings(
            "DUQ", client, quarters=2, raw_dir=tmp_path,
            known_accessions={"0001536411-26-000006"},
        )
        assert result.empty

    client.get_text.assert_not_called()
    client.get_bytes.assert_not_called()


@pytest.mark.parametrize("known_accessions", [None, set(), {"an-earlier-XML-filing"}])
def test_legacy_filings_do_not_prevent_new_xml_downloads(tmp_path, monkeypatch, known_accessions):
    filings = _filings_with_legacy_entry()
    monkeypatch.setattr(runner, "fetch_sec_company_filings", lambda *args, **kwargs: filings.copy())
    client = Mock(wraps=FakeSECClient())

    result = fetch_manager_holdings(
        "DUQ", client, quarters=2, raw_dir=tmp_path, known_accessions=known_accessions,
    )

    assert set(result["accession_number"]) == {"0001536411-26-000006"}
    client.get_text.assert_called_once_with(filings.iloc[1]["index_url"])


def test_xml_eligibility_uses_filing_date_including_cutover_day(tmp_path, monkeypatch):
    filings = _filings_with_legacy_entry().iloc[1:].copy()
    filings["report_period"] = "2013-03-31"
    filings["filing_date"] = "2013-05-20"
    filings["acceptance_datetime"] = "2013-05-20T20:00:00.000Z"
    monkeypatch.setattr(runner, "fetch_sec_company_filings", lambda *args, **kwargs: filings.copy())
    client = Mock(wraps=FakeSECClient())

    result = fetch_manager_holdings("DUQ", client, quarters=2, raw_dir=tmp_path)

    assert len(result) == 1
    client.get_text.assert_called_once()


def test_missing_information_table_in_xml_era_is_an_error(tmp_path, monkeypatch):
    filings = _filings_with_legacy_entry().iloc[1:].copy()
    filings["filing_date"] = "2013-05-20"
    monkeypatch.setattr(runner, "fetch_sec_company_filings", lambda *args, **kwargs: filings.copy())
    client = Mock(spec=FakeSECClient)
    client.get_text.return_value = "<html>No information table</html>"

    with pytest.raises(ValueError, match="Cannot locate information table"):
        fetch_manager_holdings("DUQ", client, quarters=2, raw_dir=tmp_path)


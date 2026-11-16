from unittest.mock import Mock

import pytest
import requests

from thirteenf.sec_live import SECClient


def response(status=200):
    result = requests.Response()
    result.status_code = status
    result.url = "https://www.sec.gov/test"
    result._content = b'{"ok": true}'
    return result


@pytest.fixture
def client_and_sleep(monkeypatch):
    session = Mock(spec=requests.Session)
    session.headers = {}
    client = SECClient("Example research@example.com", min_interval_seconds=0, session=session)
    sleep = Mock()
    monkeypatch.setattr("thirteenf.sec_live.time.sleep", sleep)
    return client, sleep


@pytest.mark.parametrize("error_type", [
    requests.ReadTimeout, requests.ConnectTimeout, requests.ConnectionError,
])
@pytest.mark.parametrize("method, expected", [
    ("get_text", '{"ok": true}'), ("get_bytes", b'{"ok": true}'), ("get_json", {"ok": True}),
])
def test_transport_failure_is_retried(client_and_sleep, error_type, method, expected):
    client, sleep = client_and_sleep
    client.session.get.side_effect = [error_type("temporary failure"), response()]

    assert getattr(client, method)("https://www.sec.gov/test") == expected
    assert client.session.get.call_count == 2
    for call in client.session.get.call_args_list:
        assert call.args == ("https://www.sec.gov/test",)
        assert call.kwargs == {"timeout": 30}
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize("error_type", [requests.ReadTimeout, requests.ConnectionError])
@pytest.mark.parametrize("attempts", [1, 3])
def test_exhausted_transport_retries_raise_last_error(client_and_sleep, error_type, attempts):
    client, sleep = client_and_sleep
    client.max_retries = attempts
    errors = [error_type(f"failure {attempt}") for attempt in range(attempts)]
    client.session.get.side_effect = errors

    with pytest.raises(error_type) as caught:
        client.get_text("https://www.sec.gov/test")

    assert caught.value is errors[-1]
    assert client.session.get.call_count == attempts
    assert [call.args[0] for call in sleep.call_args_list] == [2 ** i for i in range(attempts - 1)]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_http_and_transport_failures_share_retry_budget(client_and_sleep, status):
    client, sleep = client_and_sleep
    client.session.get.side_effect = [response(status), requests.ReadTimeout("temporary"), response()]

    assert client.get_json("https://www.sec.gov/test") == {"ok": True}
    assert client.session.get.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


@pytest.mark.parametrize("last_failure", [response(503), requests.ReadTimeout("final timeout")])
def test_mixed_failures_raise_final_error(client_and_sleep, last_failure):
    client, sleep = client_and_sleep
    client.session.get.side_effect = [response(429), requests.ConnectionError("reset"), last_failure]

    with pytest.raises(requests.RequestException) as caught:
        client.get_text("https://www.sec.gov/test")

    if isinstance(last_failure, requests.Response):
        assert isinstance(caught.value, requests.HTTPError)
        assert caught.value.response is last_failure
    else:
        assert caught.value is last_failure
    assert client.session.get.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


@pytest.mark.parametrize("status", [403, 404])
def test_permanent_http_error_is_not_retried(client_and_sleep, status):
    client, sleep = client_and_sleep
    client.session.get.return_value = response(status)

    with pytest.raises(requests.HTTPError):
        client.get_text("https://www.sec.gov/test")

    assert client.session.get.call_count == 1
    sleep.assert_not_called()


def test_rate_limit_applies_after_failed_attempt(client_and_sleep, monkeypatch):
    client, sleep = client_and_sleep
    client.min_interval_seconds = 2
    # First request starts at 100, fails at 101, then waits one second before retrying.
    monkeypatch.setattr("thirteenf.sec_live.time.monotonic", Mock(side_effect=[100, 101, 102, 104]))
    client.session.get.side_effect = [requests.ReadTimeout("temporary"), response()]

    assert client.get_json("https://www.sec.gov/test") == {"ok": True}
    # The backoff alone is shorter than the configured minimum interval.
    assert [call.args[0] for call in sleep.call_args_list] == [1, 1]


@pytest.mark.parametrize("attempts", [0, -1])
def test_retry_budget_must_allow_an_attempt(attempts):
    with pytest.raises(ValueError, match="max_retries"):
        SECClient("Example research@example.com", max_retries=attempts)

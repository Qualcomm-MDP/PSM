"""GET with a few retries for flaky public map services (NAIP, elevation tiles)."""

from __future__ import annotations

import time
from typing import Any

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


def get_with_retries(
    session: requests.Session,
    url: str,
    attempts: int = 4,
    backoff_s: float = 2.0,
    **kwargs: Any,
) -> requests.Response:
    """session.get(url, **kwargs), tried again after backoff_s, 2x, 4x ... on a connection
    error, a timeout or a busy/failing server (429, 5xx); the last failure is raised."""
    for attempt in range(attempts):
        try:
            r = session.get(url, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts - 1:
                raise
        else:
            if r.status_code not in RETRY_STATUS or attempt == attempts - 1:
                r.raise_for_status()
                return r
        time.sleep(backoff_s * 2**attempt)
    raise RuntimeError(f"no answer from {url}")  # not reached

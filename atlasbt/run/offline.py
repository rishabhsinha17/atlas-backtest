"""Offline guard (ARCHITECTURE §10: tests never touch the network; runs must be reproducible from cache/).

``enable(strict=False)``:
* ``atlasbt.data.http.HttpClient._get_follow`` raises ``FetchError(url, None, "offline")`` -> the client
  still serves its on-disk raw cache, and callers that already handle FetchError (e.g. the identity
  layer's SEC name lookups) degrade exactly as on a fetch failure. Blocked URLs are counted in
  ``BLOCKED`` and recorded in run provenance.
* ``strict=True`` (tests) additionally makes ``requests`` and raw sockets raise ``OfflineError`` so any
  other network path fails loudly instead of hanging.
Enabled by the runner when ATLASBT_OFFLINE != "0" (default on) and by tests/run/conftest.py. A native
``ATLASBT_OFFLINE`` switch in atlasbt.data.http was requested from the DATA owner.
"""
from __future__ import annotations

import os
import socket
from collections import Counter

BLOCKED: Counter = Counter()
_ENABLED = {"soft": False, "strict": False}


class OfflineError(RuntimeError):
    pass


def enable(strict: bool = False) -> None:
    from urllib.parse import urlparse

    from atlasbt.data import http as H

    if not _ENABLED["soft"]:
        def _offline_get_follow(self, url, params, headers, timeout, max_hops: int = 6):
            BLOCKED[urlparse(url).netloc] += 1
            raise H.FetchError(url, None, "offline (atlasbt.run.offline)")

        H.HttpClient._get_follow = _offline_get_follow
        _ENABLED["soft"] = True
    if strict and not _ENABLED["strict"]:
        import requests

        def _no_requests(self, method, url, *a, **k):
            BLOCKED[str(url)[:60]] += 1
            raise OfflineError(f"network access blocked in offline mode: {method} {url}")

        requests.sessions.Session.request = _no_requests
        _orig_connect = socket.socket.connect

        def _no_connect(self, address):
            if isinstance(address, tuple) and address and address[0] not in ("127.0.0.1", "localhost", "::1"):
                BLOCKED[str(address[0])] += 1
                raise OfflineError(f"socket connect blocked in offline mode: {address}")
            return _orig_connect(self, address)

        socket.socket.connect = _no_connect
        _ENABLED["strict"] = True


def enabled_by_env() -> bool:
    return os.environ.get("ATLASBT_OFFLINE", "1") != "0"


def blocked_summary() -> dict:
    return dict(BLOCKED)

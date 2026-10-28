"""Polite HTTP client: honest User-Agent, per-host throttle, retry with
exponential backoff, and an on-disk raw-response cache so nothing is ever
fetched twice (unless `refresh=True` or the cached copy is older than `ttl`).

Never spoofs a browser UA, never retries policy denials (403) aggressively,
never attempts to bypass bot checks.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlencode, urlparse

import requests

from . import config

log = logging.getLogger("atlasbt.data.http")


class FetchError(RuntimeError):
    def __init__(self, url: str, status: Optional[int], msg: str = ""):
        super().__init__(f"{status} {url} {msg}")
        self.url, self.status = url, status


class _HostThrottle:
    def __init__(self):
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}
        self._host_locks: dict[str, threading.Lock] = {}

    def wait(self, host: str) -> None:
        with self._lock:
            hl = self._host_locks.setdefault(host, threading.Lock())
        with hl:
            gap = config.HOST_MIN_INTERVAL.get(host, config.HOST_MIN_INTERVAL["default"])
            now = time.monotonic()
            t = self._next.get(host, 0.0)
            if t > now:
                time.sleep(t - now)
            self._next[host] = time.monotonic() + gap


_THROTTLE = _HostThrottle()


def cache_key(url: str, params: Optional[Mapping[str, Any]] = None, extra: str = "") -> str:
    full = url + ("?" + urlencode(sorted((params or {}).items())) if params else "") + extra
    return hashlib.sha1(full.encode()).hexdigest()


class HttpClient:
    def __init__(self, raw_root: Optional[Path] = None, user_agent: str = config.USER_AGENT,
                 max_retries: int = 5, timeout: float = 60.0):
        self.raw_root = Path(raw_root or config.RAW_ROOT)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.max_retries = max_retries
        self.timeout = timeout

    # ------------------------------------------------------------------
    def _paths(self, url: str, key: str) -> tuple[Path, Path]:
        host = urlparse(url).netloc.replace(":", "_")
        d = self.raw_root / host / key[:2]
        return d / f"{key}.bin", d / f"{key}.json"

    def cached(self, url: str, params=None, extra: str = "") -> Optional[bytes]:
        body, _ = self._paths(url, cache_key(url, params, extra))
        return self._read_body(body)

    def get(self, url: str, params: Optional[Mapping[str, Any]] = None, *, cache: bool = True,
            ttl: Optional[float] = None, refresh: bool = False, headers: Optional[dict] = None,
            ok_statuses: tuple[int, ...] = (200,), allow_404: bool = False,
            timeout: Optional[float] = None, cache_extra: str = "",
            validate=None) -> Optional[bytes]:
        """Return response bytes. Cached on disk (keyed by URL+params) when `cache`.

        ttl: seconds after which a cached copy is considered stale (None = forever).
        allow_404: return None (and cache the miss) on 404 instead of raising.
        validate: optional callable(bytes) -> bool; a 200 response failing it is
        returned but NOT cached (e.g. DoltHub query timeouts come back as 200).
        """
        key = cache_key(url, params, cache_extra)
        body_p, meta_p = self._paths(url, key)
        if cache and not refresh and meta_p.exists():
            meta = json.loads(meta_p.read_text())
            fresh = ttl is None or (time.time() - meta.get("fetched_epoch", 0)) < ttl
            if fresh:
                if meta.get("status") == 404:
                    return None
                b = self._read_body(body_p)
                if b is not None:
                    return b
        host = urlparse(url).netloc
        if host in config.SEC_HOSTS and config.SEC_USER_AGENT:
            headers = {**(headers or {}), "User-Agent": config.SEC_USER_AGENT}
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            _THROTTLE.wait(host)
            try:
                r = self._get_follow(url, params, headers, timeout or self.timeout)
            except requests.RequestException as e:  # network hiccup -> backoff
                last_err = e
                self._sleep(attempt)
                continue
            if r.status_code in ok_statuses:
                if cache and (validate is None or validate(r.content)):
                    self._store(body_p, meta_p, url, params, r.status_code, r.content)
                return r.content
            if r.status_code == 404 and allow_404:
                if cache:
                    self._store(None, meta_p, url, params, 404, b"")
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else None
                last_err = FetchError(url, r.status_code, r.text[:200])
                self._sleep(attempt, wait)
                continue
            raise FetchError(url, r.status_code, r.text[:300])
        raise FetchError(url, getattr(last_err, "status", None), f"gave up after retries: {last_err}")

    def _get_follow(self, url, params, headers, timeout, max_hops: int = 6) -> requests.Response:
        """GET following redirects manually, upgrading http:// targets to https://
        (only HTTPS egress is available; several .gov sites redirect to http)."""
        r = self.session.get(url, params=params, headers=headers, timeout=timeout, allow_redirects=False)
        hops = 0
        while r.is_redirect and hops < max_hops:
            loc = r.headers.get("Location", "")
            nxt = requests.compat.urljoin(r.url, loc)
            if nxt.startswith("http://"):
                nxt = "https://" + nxt[len("http://"):]
            r = self.session.get(nxt, headers=headers, timeout=timeout, allow_redirects=False)
            hops += 1
        return r

    @staticmethod
    def _sleep(attempt: int, wait: Optional[float] = None) -> None:
        time.sleep(wait if wait is not None else min(60.0, (2 ** attempt) + random.random()))

    @staticmethod
    def _read_body(body_p: Path) -> Optional[bytes]:
        gz = body_p.with_suffix(".bin.gz")
        if gz.exists():
            return gzip.decompress(gz.read_bytes())
        return body_p.read_bytes() if body_p.exists() else None

    @staticmethod
    def _store(body_p: Optional[Path], meta_p: Path, url, params, status, content: bytes) -> None:
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        if body_p is not None:
            # large bodies are stored gzip-compressed (SEC companyfacts JSON compresses ~10x)
            target = body_p.with_suffix(".bin.gz") if len(content) > 256_000 else body_p
            data = gzip.compress(content, 6) if target.suffix == ".gz" else content
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
        meta_p.write_text(json.dumps({"url": url, "params": dict(params or {}), "status": status,
                                      "fetched_epoch": time.time(),
                                      "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "bytes": len(content)}))


_DEFAULT: Optional[HttpClient] = None


def client() -> HttpClient:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = HttpClient()
    return _DEFAULT

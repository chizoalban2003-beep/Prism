"""
prism_ssrf.py
=============
Shared server-side request forgery (SSRF) guard for any code path that fetches
a URL chosen — directly or indirectly — by an LLM, a peer node, or untrusted
user input. The same predicate must be applied at the browser agent, the
federation push/announce path, and any organ that makes outbound HTTP calls,
so that we have one place to fix when (not if) a new bypass surfaces.

Default policy: reject non-http(s) schemes, reject loopback / private /
link-local / multicast / reserved IPs, reject the cloud metadata hostnames,
and **resolve DNS hostnames** before accepting them — a name that resolves
to 169.254.169.254 must be refused even if its label looks innocuous.

The DNS resolution opens a TOCTOU race with rebinding attacks (the resolver
could return a public IP here and a private IP at fetch time), but most
real-world fetchers reuse the resolution we just performed via the urllib /
httpx connection pool. Closing the race fully requires a custom resolver
plumbed into the HTTP client; we treat that as future work.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Hostnames that resolve to loopback or cloud metadata endpoints. Block by
# label so even an IPv6 ::1 trick gets refused without needing IP arithmetic.
_BLOCKED_HOSTNAMES = frozenset({
    "localhost",
    "ip6-localhost", "ip6-loopback",
    "metadata.google.internal", "metadata",
    "instance-data",
    "169.254.169.254",  # also blocked as IP below; belt-and-braces
})


def _is_blocked_ip(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_loopback: bool,
    allow_private: bool,
) -> bool:
    """Return True if *addr* should be refused under the given policy.

    Link-local (169.254/16 — cloud metadata), multicast, reserved, and
    unspecified addresses are ALWAYS blocked: there is no legitimate use
    case for fetching them from an LLM- or peer-controlled URL.
    """
    if addr.is_link_local or addr.is_multicast or addr.is_reserved or addr.is_unspecified:
        return True
    if addr.is_loopback and not allow_loopback:
        return True
    if addr.is_private and not addr.is_loopback and not allow_private:
        # is_private covers RFC 1918 LAN; loopback is a separate bucket
        # handled above so the two toggles don't entangle.
        return True
    return False


def is_safe_external_url(
    url: str,
    *,
    allow_loopback: bool = False,
    allow_private: bool = True,
) -> bool:
    """Return True if *url* is safe to fetch under the given policy.

    Defaults reflect the dominant use case in PRISM: federated peers and
    smart-home calls live on the LAN (`192.168.0.0/16`, `10.0.0.0/8`,
    `172.16.0.0/12`), so private addresses are allowed by default.
    Loopback, link-local (cloud metadata), multicast, reserved, and
    unspecified addresses are always refused unless a caller explicitly
    opts in.

    Browser-agent and LLM-chosen URL callers should pass
    ``allow_private=False`` — there is no legitimate reason for the LLM
    to scrape the user's LAN.
    """
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in _BLOCKED_HOSTNAMES:
        return False

    try:
        literal = ipaddress.ip_address(host)
        return not _is_blocked_ip(
            literal, allow_loopback=allow_loopback, allow_private=allow_private,
        )
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        logger.debug("SSRF: resolution failed for %s: %s", host, exc)
        return False
    seen = set()
    for *_pre, sockaddr in infos:
        ip = sockaddr[0]
        if ip in seen:
            continue
        seen.add(ip)
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        if _is_blocked_ip(
            addr, allow_loopback=allow_loopback, allow_private=allow_private,
        ):
            return False
    return bool(seen)

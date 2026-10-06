"""Overpass, the one network source: an Overpass QL query in, its JSON out.

Mirrors are tried in turn with retries, and the first that answers is pinned for the process so
every fetch sees one replication state. ROAD_SKETCHES_OFFLINE refuses the network outright, so a
test that reaches here has a missing fixture rather than a flaky dependency.
"""
import os

import requests


class OfflineCacheMiss(RuntimeError):
    """ROAD_SKETCHES_OFFLINE is set and a fetch wasn't satisfied from the fixture cache."""


NOMINATIM_USER_AGENT = "hopewell-road-sketches-research/0.1 (contact: rollo.l@northeastern.edu)"
OVERPASS_USER_AGENT = NOMINATIM_USER_AGENT

# The public Overpass instances are shared/rate-limited infrastructure and
# occasionally 504 under load - try a couple of mirrors with retries before
# giving up, rather than failing the whole pipeline on a transient timeout.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


_pinned_mirror: list[str] = []

# How long to wait for the TCP handshake, separately from the read. Overpass queries can
# legitimately take tens of seconds to ANSWER, but connecting is either quick or the mirror
# is not there.
CONNECT_TIMEOUT_S = 5

# Mirrors that have already failed their whole retry budget this run.
_dead_mirrors: set[str] = set()


def query_overpass(query: str, attempts_per_mirror: int = 4, timeout: int = 30) -> dict:
    """POST an Overpass QL query, retrying across mirrors on timeout/5xx errors.

    Once a mirror answers, it is PINNED for the rest of the process, so every fetch sees
    the same snapshot. Retries are per-mirror before failover.
    """
    if os.environ.get("ROAD_SKETCHES_OFFLINE"):
        # The test suite runs against a committed fixture cache. If something reaches this
        # far it means the fixture is missing, and the honest outcome is a loud failure -
        # not a silent network call that makes the tests depend on Overpass's uptime and
        # current replication state.
        raise OfflineCacheMiss(
            "ROAD_SKETCHES_OFFLINE is set and this query is not in the fixture cache. Add the "
            "response to tests/fixtures/osm_cache (see tests/conftest.py) rather than "
            f"letting a test reach the network. Query was:\n{query.strip()[:400]}")

    ordered = ([m for m in _pinned_mirror if m in OVERPASS_MIRRORS]
               + [m for m in OVERPASS_MIRRORS if m not in _pinned_mirror])
    last_error = None
    for mirror in ordered:
        host = mirror.split("//")[1].split("/")[0]
        if host in _dead_mirrors:
            continue  # already proved unreachable this run; don't pay the timeout again
        for attempt in range(attempts_per_mirror):
            try:
                resp = requests.post(
                    mirror, data={"data": query}, headers={"User-Agent": OVERPASS_USER_AGENT},
                    # (connect, read). A mirror that is blackholing packets never completes
                    # the TCP handshake, and a single 30 s number spent the whole budget
                    # there: one editing session sat 10+ minutes in SYN_SENT against
                    # overpass.kumi.systems with nothing printed. A connect is either fast
                    # or not happening; a read legitimately takes a while.
                    timeout=(CONNECT_TIMEOUT_S, timeout),
                )
                resp.raise_for_status()
                if not _pinned_mirror:
                    _pinned_mirror.append(mirror)
                    print(f"  Overpass: using {host} "
                          f"(pinned for this run so every fetch sees one replication state)")
                return resp.json()
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                last_error = e
                # Say so as it happens. A retry loop that prints nothing is indistinguishable
                # from a hang, which is exactly how this failure presented.
                print(f"  Overpass: {host} attempt {attempt + 1}/{attempts_per_mirror} failed "
                      f"({type(e).__name__}); {'retrying' if attempt + 1 < attempts_per_mirror else 'giving up on it'}",
                      flush=True)
                if isinstance(e, requests.exceptions.ConnectTimeout):
                    break  # unreachable, not busy - further attempts just burn the budget
        _dead_mirrors.add(host)
    raise RuntimeError(f"All Overpass mirrors failed after retries. Last error: {last_error}")

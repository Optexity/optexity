"""Parse comma-separated ``PROXY_*`` env fields and pick one at random.

Each of ``PROXY_URL``, ``PROXY_USERNAME``, ``PROXY_PASSWORD``,
``PROXY_PROVIDER``, and ``PROXY_COUNTRY`` may be a single value or a
comma-separated list. ``PROXY_URL`` defines how many configs there are;
other fields broadcast a single value across all URLs, or align by index.
"""

from __future__ import annotations

import random
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

ProxyProvider = Literal["oxylabs", "brightdata", "other"]


@dataclass(frozen=True)
class ProxyChoice:
    url: str
    username: str | None
    password: str | None
    provider: ProxyProvider | None
    country: str | None


def _csv_parts(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _align(parts: list[str], n: int) -> list[str | None]:
    if n <= 0:
        return []
    if not parts:
        return [None] * n
    if len(parts) == 1:
        return [parts[0]] * n
    if len(parts) < n:
        return parts + [parts[-1]] * (n - len(parts))
    return parts[:n]


def _as_provider(raw: str | None) -> ProxyProvider | None:
    if raw in ("oxylabs", "brightdata", "other"):
        return raw  # type: ignore[return-value]
    return None


def proxy_configs_from_settings(settings: Any) -> list[ProxyChoice]:
    urls = _csv_parts(getattr(settings, "PROXY_URL", None))
    if not urls:
        return []
    n = len(urls)
    usernames = _align(_csv_parts(getattr(settings, "PROXY_USERNAME", None)), n)
    passwords = _align(_csv_parts(getattr(settings, "PROXY_PASSWORD", None)), n)
    providers = _align(_csv_parts(getattr(settings, "PROXY_PROVIDER", None)), n)
    countries = _align(_csv_parts(getattr(settings, "PROXY_COUNTRY", None)), n)
    configs: list[ProxyChoice] = []
    for i, url in enumerate(urls):
        provider = _as_provider(providers[i])
        country = countries[i]
        if provider == "oxylabs" and country is None:
            country = "US"
        configs.append(
            ProxyChoice(
                url=url,
                username=usernames[i],
                password=passwords[i],
                provider=provider,
                country=country,
            )
        )
    return configs


def pick_random_proxy(settings: Any) -> ProxyChoice | None:
    configs = proxy_configs_from_settings(settings)
    if not configs:
        return None
    return random.choice(configs)


@dataclass
class ProxyAssignment:
    """The ``PROXY_URL`` entry opcloud assigned to the running task, and whether
    any of its clients actually went through the proxy."""

    url: str | None
    used: bool = False


# Set per task so portal code that builds its own clients needs no task plumbing.
PROXY_ASSIGNMENT: ContextVar[ProxyAssignment | None] = ContextVar(
    "proxy_assignment", default=None
)


def pick_proxy(settings: Any, url: str | None = None) -> ProxyChoice | None:
    """The assigned entry (``url``, else the running task's), else a random one.

    Call it only when a client is about to use the proxy: it marks the task's
    assignment as used.
    """
    configs = proxy_configs_from_settings(settings)
    if not configs:
        return None
    assignment = PROXY_ASSIGNMENT.get()
    if assignment is not None:
        assignment.used = True
    wanted = url or (assignment.url if assignment is not None else None)
    return next((c for c in configs if c.url == wanted), None) or random.choice(configs)

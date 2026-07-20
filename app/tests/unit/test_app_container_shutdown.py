from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from local_llm_chat.bootstrap import _close_services_in_order


def close_action(
    name: str,
    calls: list[str],
    *,
    error: Exception | None = None,
) -> Callable[[], Awaitable[None]]:
    async def close() -> None:
        calls.append(name)
        if error is not None:
            raise error

    return close


@pytest.mark.asyncio
async def test_shutdown_attempts_every_close_and_aggregates_failures() -> None:
    calls: list[str] = []
    services = (
        ("scheduler", close_action("scheduler", calls, error=RuntimeError("one"))),
        ("memory", close_action("memory", calls)),
        ("translation", close_action("translation", calls, error=ValueError("two"))),
        ("providers", close_action("providers", calls)),
    )

    with pytest.raises(ExceptionGroup) as captured:
        await _close_services_in_order(services)

    assert calls == ["scheduler", "memory", "translation", "providers"]
    assert [type(error) for error in captured.value.exceptions] == [
        RuntimeError,
        ValueError,
    ]


@pytest.mark.asyncio
async def test_shutdown_times_out_one_close_and_still_attempts_the_rest() -> None:
    calls: list[str] = []
    never_finishes = asyncio.Event()

    async def hanging_close() -> None:
        calls.append("hanging")
        await never_finishes.wait()

    services = (
        ("hanging", hanging_close),
        ("providers", close_action("providers", calls)),
    )

    with pytest.raises(ExceptionGroup) as captured:
        await asyncio.wait_for(
            _close_services_in_order(services, timeout_seconds=0.01),
            timeout=0.2,
        )

    assert calls == ["hanging", "providers"]
    assert [type(error) for error in captured.value.exceptions] == [TimeoutError]

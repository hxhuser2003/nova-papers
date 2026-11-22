import asyncio
import pytest
from backend.chat_admission import ChatAdmission, QueueFull


def _run(coro):
    # Not asyncio.run(): it explicitly clears the main thread's event loop
    # on exit (sets it to None), which then breaks any later test in the
    # same pytest session that expects an ambient loop to already exist
    # (e.g. FastAPI's TestClient/anyio portal in test_main.py) with
    # "RuntimeError: There is no current event loop in thread 'MainThread'".
    # Running on an explicit loop and leaving it installed as current avoids
    # that cross-test pollution.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(loop)


def test_bounded_queue_timeout_and_recovery():
    async def scenario():
        gate=ChatAdmission(1,1,.02)
        await gate.acquire()
        waiter=asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)
        with pytest.raises(QueueFull):
            await gate.acquire()
        with pytest.raises(asyncio.TimeoutError):
            await waiter
        assert gate.waiting==0
        gate.release()
        await gate.acquire()
        gate.release()
    _run(scenario())


def test_cancelled_waiter_does_not_leak_capacity():
    async def scenario():
        gate=ChatAdmission(1,1,1)
        await gate.acquire()
        waiter=asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert gate.waiting==0
        gate.release()
        await gate.acquire()
        gate.release()
    _run(scenario())

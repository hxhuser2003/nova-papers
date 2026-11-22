"""Per-process bounded admission. Call only from the application's event loop."""
import asyncio


class QueueFull(Exception):
    pass


class ChatAdmission:
    def __init__(self, capacity, max_waiting, timeout):
        self.semaphore = asyncio.Semaphore(capacity)
        self.max_waiting = max_waiting
        self.timeout = timeout
        self.waiting = 0

    async def acquire(self):
        if not self.semaphore.locked():
            await self.semaphore.acquire()
            return
        if self.waiting >= self.max_waiting:
            raise QueueFull()
        self.waiting += 1
        try:
            await asyncio.wait_for(self.semaphore.acquire(), timeout=self.timeout)
        finally:
            self.waiting -= 1

    def release(self):
        self.semaphore.release()

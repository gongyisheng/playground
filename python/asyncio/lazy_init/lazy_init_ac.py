import asyncio
from lazy_init_pq import PageQueue


class AttachmentClient(object):
    def __init__(self):
        self._sift_attachment_queue = None

    @property
    def sift_attachment_queue(self):
        if self._sift_attachment_queue is None:
            self._sift_attachment_queue = PageQueue()
        return self._sift_attachment_queue

    async def add_message(self):
        await asyncio.sleep(0.1)
        await self.sift_attachment_queue.lpush()
        await asyncio.sleep(0.1)

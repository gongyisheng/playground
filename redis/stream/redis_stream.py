import asyncio
import json
import logging
import snappy

import signal_state_aio as signal_state

import lib_settings

# redis config
# config set appendonly yes
# config set appendfsync always

# Generate random data
data_size = 10
group_name = "default_group"
consumer_name = "consumer_0"
min_idle_time = 5
block_timeout = 5


class RedisStream:
    def __init__(
        self,
        redis,
        stream_name,
        group_name="default_group",
        consumer_name="default_consumer",
        pickle=json,
        compress=snappy,
    ):
        self._redis = redis
        self._stream_name = stream_name
        self._group_name = group_name
        self._consumer_name = consumer_name
        self._pickle = pickle
        self._compress = compress

    def _encode(self, msg):
        if self._pickle:
            try:
                msg = self._pickle.dumps(msg)
            except Exception as ex:
                logging.error("Json dumps error - error[%s]", ex, exc_info=True)
                raise

        if self._compress:
            msg = self._compress.compress(msg)
        return msg

    def _decode(self, msg):
        if self._compress:
            try:
                msg = self._compress.decompress(msg)
            except:
                pass
        if self._pickle:
            msg = self._pickle.loads(msg)
        return msg

    async def init(self):
        # check if stream exists
        has_stream = await self._redis.exists(self._stream_name) == 1
        if not has_stream:
            logging.info(
                f"Stream not exists: [{self._stream_name}], will create it with group"
            )

        # check if group exists
        has_group = False
        if has_stream:
            for group in resp:
                if group["name"].decode("utf-8") == self._group_name:
                    has_group = True
                    logging.info(f"Group already exists: [{self._group_name}]")
                    break

        # create group if not exists
        if not (has_group & has_stream):
            logging.info(
                f"Create group [{self._group_name}] for stream [{self._stream_name}]"
            )
            resp = await self._redis.xgroup_create(
                self._stream_name, self._group_name, id="$", mkstream=True
            )
            if resp != True:
                logging.error(f"Create group failed, response=[{resp}]")

        # check if consumer exists
        has_consumer = False
        resp = await self._redis.xinfo_consumers(self._stream_name, self._group_name)
        for consumer in resp:
            if consumer["name"].decode("utf-8") == self._consumer_name:
                has_consumer = True
                logging.info(f"Consumer already exists: [{self._consumer_name}]")
                break

        # create consumer if not exists
        if not has_consumer:
            logging.info(f"Create consumer: [{self._consumer_name}]")
            resp = await self._redis.xgroup_createconsumer(
                self._stream_name, self._group_name, self._consumer_name
            )
            if resp != 1:
                logging.error(f"Create consumer failed, response=[{resp}]")
        logging.info(
            f"Redis stream client initialized. Stream=[{self._stream_name}], Group=[{self._group_name}], Consumer=[{self._consumer_name}]"
        )

    async def claim_ownership(self):
        await self._redis.xclaim(
            self._stream_name,
            self._group_name,
            self._consumer_name,
            1,
            ["0-0"],
        )

    async def batch_put(self, values=[]):
        _buffer = []
        for item in values:
            _buffer.append(self._encode(item))

        need_retry = True
        while need_retry:
            try:
                pipe = self._redis.pipeline()
                for item in _buffer:
                    pipe.xadd(self._stream_name, {"data": item})
                await pipe.execute()
                need_retry = False
            except Exception as ex:
                logging.error("Batch send error - error[%s]", ex, exc_info=True)
                pipe.reset()
                await asyncio.sleep(1)

            if not signal_state.ALIVE:
                break

        logging.info(
            f"Batch put - Send {len(_buffer)} messages to stream [{self._stream_name}], group=[{self._group_name}]"
        )

    async def batch_get(self, count=10, block=5):
        _buffer = []
        res = await self._redis.xreadgroup(
            groupname=self._group_name,
            consumername=self._consumer_name,
            streams={self._stream_name: ">"},
            count=count,
            block=block,
        )
        if len(res) == 0:
            logging.info("Batch get - No data to read")
            return
        stream_data = res[0][1]
        for item in stream_data:
            id = item[0].decode("utf-8")
            value = self._decode(item[1][b"data"])
            _buffer.append((id, value))

        logging.info(
            f"Batch get - Get {len(stream_data)} messages from stream [{self._stream_name}], group=[{self._group_name}]"
        )
        return _buffer

    async def batch_ack(self, successful_ids):
        if len(successful_ids) == 0:
            logging.info("Batch ack - No data to ack")

        need_retry = True
        while need_retry:
            try:
                await self._redis.xack(
                    self._stream_name, self._group_name, *successful_ids
                )
                logging.info(
                    f"Batch ack - Acked {len(successful_ids)} messages in stream=[{self._stream_name}], group=[{self._group_name}]"
                )
                need_retry = False
            except Exception as ex:
                logging.error("Batch ack error - error[%s]", ex, exc_info=True)
                await asyncio.sleep(1)

            if not signal_state.ALIVE:
                break

    async def get_lag(self):
        resp = await self._redis.xpending(self._stream_name, self._group_name)
        return resp

    async def process(self, buffer):
        pass

    async def run(self):
        await self.init()
        while signal_state.ALIVE:
            try:
                # batch get
                _buffer = await self.batch_get()
                if _buffer:
                    successful_ids = self.process(_buffer)
                    await self.batch_ack(successful_ids)
                else:
                    await asyncio.sleep(1)
            except Exception as ex:
                logging.error("Run error - error[%s]", ex, exc_info=True)
                await asyncio.sleep(1)

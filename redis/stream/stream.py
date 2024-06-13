import asyncio
import json
import logging
import snappy
import time
from typing import List, Optional, Tuple, Union

import redis.asyncio as aioredis
import signal_state_aio as signal_state

# redis config
# config set appendonly yes
# config set appendfsync always

# redis client config
# decode_responses=False: must return bytes


def ensure_str(s: Union[str, bytes]) -> str:
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return s


def ensure_bytes(s: Union[str, bytes]) -> bytes:
    if isinstance(s, str):
        return s.encode("utf-8")
    return s


# KNOWN ISSUE
# 1. This package is tested under redis=6.2.14. XAUTOCLAIM may not work as expected in redis >= 6.2.14 due to API change.
class RedisStream:
    MESSAGE_ID_KEY = "MESSAGE_ID"
    MESSAGE_DATA_KEY = "MESSAGE_DATA"

    def __init__(
        self,
        redis: aioredis.Redis,
        stream_name: str,
        group_name: str = "default_group",
        consumer_name: str = "default_consumer",
        pickle=json,
        compress=snappy,
    ) -> None:
        self._stream_name = stream_name
        self._group_name = group_name
        self._consumer_name = consumer_name
        self._pickle = pickle
        self._compress = compress
        self._redis = redis

    def _encode(self, msg: dict) -> bytes:
        """
        Dump and compress message to bytes before sending to redis
        """
        if self._pickle:
            try:
                msg = self._pickle.dumps(msg)
            except Exception as ex:
                logging.error("Json dumps error - error[%s]", ex, exc_info=True)
                raise

        if self._compress:
            msg = self._compress.compress(msg)
        return msg

    def _decode(self, msg: bytes) -> dict:
        """
        Decompress and load message to dict after receiving from redis
        """
        if self._compress:
            try:
                msg = self._compress.decompress(msg)
            except:
                pass
        if self._pickle:
            msg = self._pickle.loads(msg)
        return msg

    async def init(self) -> None:
        """
        Initialize redis stream client
        Create stream, group and consumer if not exists
        """
        # check if stream exists
        has_stream = await self._redis.exists(self._stream_name) == 1
        if not has_stream:
            logging.info(
                f"Stream not exists: [{self._stream_name}], will create it with group"
            )

        # check if group exists
        has_group = False
        if has_stream:
            resp = await self._redis.xinfo_groups(self._stream_name)
            for group in resp:
                if ensure_str(group["name"]) == self._group_name:
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
            if resp is not True:
                logging.error(f"Create group failed, response=[{resp}]")

        # check if consumer exists
        has_consumer = False
        resp = await self._redis.xinfo_consumers(self._stream_name, self._group_name)
        for consumer in resp:
            if ensure_str(consumer["name"]) == self._consumer_name:
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

    async def batch_put(
        self, values: List[str] = [], retry: Optional[int] = None
    ) -> bool:
        """
        Put a batch of messages to redis stream using pipeline
        :param values: list of messages to put
        :param retry: number of retries if failed
        :return: True if success, False if failed

        If failed, retry until success or retry times reached
        If exit signal received, stop retry and exit
        Only support string type messages, bytes type messages should be converted to string before calling this method
        """
        _buffer = []
        for item in values:
            _buffer.append(self._encode(item))

        if retry is None:
            retry = float("inf")

        succ = False
        while not succ and retry > 0:
            try:
                pipe = self._redis.pipeline()
                for item in _buffer:
                    pipe.xadd(self._stream_name, {"data": item})
                await pipe.execute()

                succ = True
                logging.info(
                    f"Batch put - Send {len(_buffer)} messages to stream [{self._stream_name}], group=[{self._group_name}]"
                )
                break
            except Exception as ex:
                logging.error("Batch send error - error[%s]", ex, exc_info=True)
                await pipe.reset()
                await asyncio.sleep(1)

            retry -= 1
            if not signal_state.ALIVE:
                logging.info("Batch send - exit signal received")
                break

        return succ

    async def batch_get(self, count: int = 10, block: int = 5000) -> List[dict]:
        """
        Batch get messages from redis stream
        :param count: number of messages to get
        :param block: block time in milliseconds
        :return: list of messages

        This method will block until messages received or timeout
        This method may raise exception if error occurred
        If available messages number less than count, return all available messages
        If no message received, return empty list
        """
        _buffer = []
        res = await self._redis.xreadgroup(
            groupname=self._group_name,
            consumername=self._consumer_name,
            streams={self._stream_name: ">"},
            count=count,
            block=block,
        )
        if len(res) == 0:
            logging.info(
                f"Batch get - No data to read from stream [{self._stream_name}], group=[{self._group_name}]"
            )
            return _buffer

        stream_data = res[0][1]
        for item in stream_data:
            id = ensure_str(item[0])
            value = self._decode(item[1][ensure_bytes("data")])
            _buffer.append({self.MESSAGE_ID_KEY: id, self.MESSAGE_DATA_KEY: value})

        message_ids = [item[self.MESSAGE_ID_KEY] for item in _buffer]
        logging.info(
            f"Batch get - Get {len(stream_data)} messages from stream [{self._stream_name}], group=[{self._group_name}], message_ids={message_ids}"
        )
        return _buffer

    async def batch_claim(self, count: int = 10) -> None:
        """
        Auto claim ownership for current <stream, group, consumer>
        Get messages from pending list
        :param count: number of messages to claim

        This method will claim ownership for messages in pending list
        After claiming, messages in pending list will be re-delivered to consumer
        """
        _buffer = []
        resp = await self._redis.xautoclaim(
            self._stream_name,
            self._group_name,
            self._consumer_name,
            0,
            count=count,
        )

        # Build stream_data and delete_data based on response
        if len(resp) > 2:
            # redis > 6.2.14
            stream_data = resp[1]
            delete_data = resp[2]
        else:
            # redis <= 6.2.14
            stream_data = []
            delete_data = []
            for item in resp[1]:
                if item[0] is None:
                    delete_data.append(item)
                else:
                    stream_data.append(item)

        if len(delete_data) > 0:
            logging.error(
                f"Batch claim - Found {len(delete_data)} deleted messages in pending list of stream=[{self._stream_name}], group=[{self._group_name}], consumer=[{self._consumer_name}]. delete_data={delete_data}"
            )

        if len(stream_data) == 0:
            logging.info(
                f"Batch claim - No data to claim from pending list of stream=[{self._stream_name}], group=[{self._group_name}], consumer=[{self._consumer_name}]"
            )
            return _buffer

        for item in stream_data:
            id = ensure_str(item[0])
            value = self._decode(item[1][ensure_bytes("data")])
            _buffer.append({self.MESSAGE_ID_KEY: id, self.MESSAGE_DATA_KEY: value})

        message_ids = [item[self.MESSAGE_ID_KEY] for item in _buffer]
        logging.info(
            f"Batch claim - Claimed {len(stream_data)} messages from pending list of stream=[{self._stream_name}], group=[{self._group_name}], consumer=[{self._consumer_name}], message_ids={message_ids}"
        )
        return _buffer

    async def batch_ack(
        self, successful_ids: List[str] = [], retry: Optional[int] = None
    ) -> bool:
        """
        Batch ack messages in redis stream
        :param successful_ids: list of message ids to ack
        :return: True if success, False if failed

        If failed, retry until success or retry times reached
        If exit signal received, stop retry and exit
        """
        if len(successful_ids) == 0:
            logging.info("Batch ack - No data to ack")

        if retry is None:
            retry = float("inf")

        succ = False
        while not succ and retry > 0:
            try:
                await self._redis.xack(
                    self._stream_name, self._group_name, *successful_ids
                )

                succ = True
                logging.info(
                    f"Batch ack - Acked {len(successful_ids)} messages in stream=[{self._stream_name}], group=[{self._group_name}]"
                )
                break
            except Exception as ex:
                logging.error("Batch ack error - error[%s]", ex, exc_info=True)
                await asyncio.sleep(1)

            retry -= 1
            if not signal_state.ALIVE:
                logging.info("Batch ack - exit signal received")
                break

        return succ

    async def monitor_xpel(self) -> Tuple[int, int]:
        """
        Get pending list length and min timestamp for current <stream, group>
        :return: pending list length
        """
        resp = await self._redis.xpending(self._stream_name, self._group_name)
        length = resp["pending"]
        min_id = ensure_str(resp["min"])
        max_id = ensure_str(resp["max"])

        min_time = -1
        try:
            min_time = int(min_id.split("-")[0]) // 1000
        except Exception as ex:
            logging.error("Monitor xpel - Get min time error[%s]", ex, exc_info=True)
        logging.info(
            f"Get pending list length - Length: {length}, Min ID: {min_id}, Max ID: {max_id}, Min Time: {min_time}"
        )

        return length, min_time

    async def trim(self, ttl: int, approximate: bool = True) -> Tuple[int, int]:
        """
        Trim stream based on TTL (time-to-live)
        :param ttl: time-to-live in seconds
        :return: number of messages deleted
        """
        minid = f"{int((time.time() - ttl)*1000)}-0"
        pipe = self._redis.pipeline()
        pipe.xtrim(self._stream_name, minid=minid, approximate=approximate)
        pipe.xlen(self._stream_name)
        delete_count, stream_length = await pipe.execute()
        logging.info(
            f"Trim - Trim stream=[{self._stream_name}], ttl=[{ttl}], minid=[{minid}], delete_count=[{delete_count}], stream_length=[{stream_length}]"
        )
        return delete_count, stream_length

    async def range_replay(
        self, start_id: str, end_id: str, count: int = 10, left_closed: bool = True
    ) -> List[dict]:
        """
        Replay messages in a time range from start_time to end_time
        :param start_id: start id
        :param end_id: end id
        :param count: number of messages to process each time
        :param left_closed: whether to include messages with id = start_id or not
        """
        if not left_closed:
            start_id = f"({start_id}"
        _buffer = []
        resp = await self._redis.xrange(
            self._stream_name, start_id, end_id, count=count
        )

        if len(resp) == 0:
            logging.info(
                f"Batch replay - No data to replay from stream=[{self._stream_name}], start_id=[{start_id}], end_id=[{end_id}], count=[{count}], left_closed=[{left_closed}]"
            )
            return _buffer

        for item in resp:
            id = ensure_str(item[0])
            value = self._decode(item[1][ensure_bytes("data")])
            _buffer.append({self.MESSAGE_ID_KEY: id, self.MESSAGE_DATA_KEY: value})

        max_id = _buffer[-1][self.MESSAGE_ID_KEY]
        message_ids = [item[self.MESSAGE_ID_KEY] for item in _buffer]
        logging.info(
            f"Batch replay - Replay messages from stream=[{self._stream_name}], start_id=[{start_id}], end_id=[{end_id}], count=[{count}], left_closed=[{left_closed}, max_id=[{max_id}], message_ids={message_ids}"
        )
        return _buffer

    async def item_replay(self, message_ids: List[str]) -> List[dict]:
        """
        Replay messages by specific message ids
        :param message_ids: list of message ids
        :return: list of messages
        """
        _buffer = []
        pipe = self._redis.pipeline()
        for id in message_ids:
            pipe.xrange(self._stream_name, min=id, max=id)
        resp = await pipe.execute()
        resp = [item[0] for item in resp if len(item) > 0]

        if len(resp) == 0:
            logging.info(
                f"Item replay - No data to replay from stream=[{self._stream_name}], message_ids={message_ids}"
            )
            return _buffer

        for item in resp:
            id = ensure_str(item[0])
            value = self._decode(item[1][ensure_bytes("data")])
            _buffer.append({self.MESSAGE_ID_KEY: id, self.MESSAGE_DATA_KEY: value})

        logging.info(
            f"Item replay - Replay messages from stream=[{self._stream_name}], message_ids={message_ids}"
        )
        return _buffer

    async def batch_process(self, buffer: List[dict]) -> List[str]:
        """
        Process messages in buffer and return successful message ids
        :param buffer: list of messages to process
        :return: list of successful message ids

        This method should be implemented by subclass
        If this method raises exception, the whole batch will not be acked
        If this method returns partial successful message ids, the rest will not be acked
        """
        raise NotImplementedError

    async def run(self, count: int = 10, block: int = 5000) -> None:
        """
        Run the real-time stream client
        :param count: number of messages to process each time
        :param block: block time in milliseconds

        This method will claim messages from pending list first, process them, then ack.
        If no pending messages, get messages from stream and process them, then ack.
        If no messages available, sleep for 1s and retry.
        If exit signal received, stop and exit after processing current batch.
        """
        await self.init()
        need_claim = True
        while signal_state.ALIVE:
            try:
                if need_claim:
                    # claim message for pending messages
                    _buffer = await self.batch_claim(count=count)
                    if len(_buffer) == 0:
                        need_claim = False
                        continue
                else:
                    # get message from stream
                    _buffer = await self.batch_get(count=count, block=block)
                    if len(_buffer) == 0:
                        await asyncio.sleep(1)
                        continue

                successful_ids = await self.batch_process(_buffer)
                await self.batch_ack(successful_ids)
            except Exception as ex:
                logging.error("Run - Get error[%s]", ex, exc_info=True)
                await asyncio.sleep(1)
        logging.info("Run - Exit redis stream client loop")

    async def run_range_replay(
        self, start_time: int, end_time: int, count: int = 10
    ) -> None:
        """
        Run the replay client
        :param start_time: start time to replay
        :param end_time: end time to replay
        :param count: number of messages to process each time

        This method will replay messages from start_time to end_time, process them.
        No ack will be done in this method.
        If no messages available, exit the loop.
        If exit signal received, stop and exit after processing current batch.
        """
        if start_time is None or end_time is None:
            logging.error("Run range replay error - Start time or end time is None")
            return
        elif start_time >= end_time:
            logging.error(
                f"Run range replay error - Start time [{start_time}] is greater than end time [{end_time}]"
            )
            return
        elif start_time < 0 or end_time < 0:
            logging.error(
                f"Run range replay error - Start time [{start_time}] or end time [{end_time}] is less than 0"
            )
            return

        await self.init()
        start_id = f"{start_time*1000}-0"
        end_id = f"{end_time*1000}-0"
        left_closed = True

        while signal_state.ALIVE:
            try:
                # replay messages from stream
                _buffer = await self.range_replay(
                    start_id, end_id, count=count, left_closed=left_closed
                )
                if len(_buffer) == 0:
                    break

                await self.batch_process(_buffer)
                start_id = _buffer[-1][self.MESSAGE_ID_KEY]
                left_closed = False
            except Exception as ex:
                logging.error("Run range replay - Get error[%s]", ex, exc_info=True)
                await asyncio.sleep(1)
        logging.info("Run range replay - Exit redis stream client loop")

    async def run_item_replay(self, message_ids: List[str]) -> None:
        """
        Run the replay client
        :param message_ids: list of message ids to replay

        This method will replay messages by message ids, process them.
        No ack will be done in this method.
        If no messages available, exit the loop.
        If exit signal received, stop and exit after processing current batch.
        """
        if message_ids is None:
            logging.error("Run item replay error - Message ids is None")
            return
        elif len(message_ids) == 0:
            logging.error("Run item replay error - Message ids is empty")
            return

        await self.init()
        cutoff = 0
        while signal_state.ALIVE:
            try:
                _batch_message_ids = message_ids[cutoff : cutoff + 10]
                # replay messages from stream
                _buffer = await self.item_replay(_batch_message_ids)
                if len(_buffer) == 0:
                    break

                await self.batch_process(_buffer)

                if cutoff >= len(message_ids):
                    break
                cutoff += 10
            except Exception as ex:
                logging.error("Run item replay - Get error[%s]", ex, exc_info=True)
                await asyncio.sleep(1)
        logging.info("Run item replay - Exit redis stream client loop")

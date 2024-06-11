from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import (
    DeleteRowsEvent,
    UpdateRowsEvent,
    WriteRowsEvent,
)

import time
import sys
from config import MYSQL_SETTINGS

# Before running this script, you need to run the following command in mysql:
# SET GLOBAL binlog_row_metadata = "FULL"
# SET GLOBAL binlog_row_image = "FULL"

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else None
LOG_POS = int(sys.argv[2]) if len(sys.argv) > 2 else None
RESUME_STREAM = LOG_FILE is not None and LOG_POS is not None

QPS_DICT = {}

# async def print_qps():
#     while True:
#         now = int(time.time())
#         print(f"{now-1} QPS: {QPS_DICT.get(now-1, 0)}")
#         await asyncio.sleep(1)


def binlog_subscribe():
    # server_id is your slave identifier, it should be unique.
    # set blocking to True if you want to block and wait for the next event at
    # the end of the stream
    global QPS_DICT
    stream = BinLogStreamReader(
        connection_settings=MYSQL_SETTINGS,
        server_id=2,
        only_events=[DeleteRowsEvent, UpdateRowsEvent, WriteRowsEvent],
        resume_stream=RESUME_STREAM,
        blocking=False,
        enable_logging=True,
        only_schemas={"cdc_test"},
        only_tables={"test"},
        log_file=LOG_FILE,
        log_pos=LOG_POS,
    )
    print("Start to subscribe binlog")
    try:
        while True:
            binlogevent = stream.fetchone()
            if binlogevent is None:
                print("No event, retry...")
                time.sleep(1)
                continue
            db = binlogevent.schema
            table = binlogevent.table
            timestamp = binlogevent.timestamp
            log_file = stream.log_file
            log_pos = stream.log_pos
            print("-----Event-----")
            for row in binlogevent.rows:
                if timestamp not in QPS_DICT:
                    QPS_DICT[timestamp] = {"insert": 0, "update": 0, "delete": 0}
                if isinstance(binlogevent, DeleteRowsEvent):
                    print("Delete Event")
                    message_body = row["values"]
                    QPS_DICT[timestamp]["delete"] += 1
                elif isinstance(binlogevent, UpdateRowsEvent):
                    print("Update Event")
                    message_body = row["after_values"]
                    QPS_DICT[timestamp]["update"] += 1
                elif isinstance(binlogevent, WriteRowsEvent):
                    print("Insert Event")
                    message_body = row["values"]
                    QPS_DICT[timestamp]["insert"] += 1
                print(
                    f"[{log_file}-{log_pos}][{db}.{table}] time={timestamp}, body={message_body}"
                )

    except KeyboardInterrupt:
        stream.close()


if __name__ == "__main__":
    binlog_subscribe()
    for k in sorted(list(QPS_DICT.keys())):
        print(f"{k} QPS: {QPS_DICT[k]}")

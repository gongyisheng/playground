from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import (
    DeleteRowsEvent,
    UpdateRowsEvent,
    WriteRowsEvent,
)
from config import MYSQL_SETTINGS

# Before running this script, you need to run the following command in mysql:
# SET GLOBAL binlog_row_metadata = "FULL"
# SET GLOBAL binlog_row_image = "FULL"

def main():
    # server_id is your slave identifier, it should be unique.
    # set blocking to True if you want to block and wait for the next event at
    # the end of the stream
    stream = BinLogStreamReader(
        connection_settings=MYSQL_SETTINGS, 
        server_id=3, 
        only_events=[DeleteRowsEvent, UpdateRowsEvent, WriteRowsEvent],
        blocking=True,
    )
    try:
        for binlogevent in stream:
            db = binlogevent.schema
            table = binlogevent.table
            time = binlogevent.timestamp
            log_file = stream.log_file
            log_pos = stream.log_pos
            for row in binlogevent.rows:
                if isinstance(binlogevent, DeleteRowsEvent):
                    print("Delete Event")
                    message_body = row["values"]
                elif isinstance(binlogevent, UpdateRowsEvent):
                    print("Update Event")
                    message_body = row["after_values"]
                elif isinstance(binlogevent, WriteRowsEvent):
                    print("Insert Event")
                    message_body = row["values"]
                print(f"[{log_file}-{log_pos}][{db}.{table}] time={time}, body={message_body}")
    except KeyboardInterrupt:
        stream.close()

if __name__ == "__main__":
    main()
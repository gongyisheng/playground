import datetime
import pymysql

MYSQL_CONFIG = dict(host="localhost", port=3306, user="mysql", password="mysql", database="langgraph")
RETENTION_SECONDS = 30 * 24 * 60 * 60

# checkpoint column has a "ts" field with ISO timestamp
FIND_STALE_THREADS = """
SELECT thread_id FROM checkpoints
GROUP BY thread_id
HAVING MAX(JSON_UNQUOTE(JSON_EXTRACT(checkpoint, '$.ts'))) < %s
"""

DELETE_QUERIES = [
    "DELETE FROM checkpoint_writes WHERE thread_id = %s",
    "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
    "DELETE FROM checkpoints WHERE thread_id = %s",
]


def cleanup():
    cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=RETENTION_SECONDS)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")
    conn = pymysql.connect(**MYSQL_CONFIG, autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(FIND_STALE_THREADS, (cutoff_str,))
            stale_threads = [row[0] for row in cur.fetchall()]

            if not stale_threads:
                print("no stale threads found")
                return

            for tid in stale_threads:
                for q in DELETE_QUERIES:
                    cur.execute(q, (tid,))

            conn.commit()
            print(f"deleted {len(stale_threads)} threads: {stale_threads}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    cleanup()

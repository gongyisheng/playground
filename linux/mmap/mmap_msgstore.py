"""mmap message store — Python version of mmap_msgstore.c"""
import mmap
import struct
import os
import time

DB_FILE = "/tmp/mmap_msgstore_py.db"
MAX_FILESIZE = 1024 * 1024  # 1MB

# header: msg_id(Q=uint64) + length(I=uint32) + timestamp(Q=uint64) = 20 bytes
HEADER_FORMAT = "QIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class MsgStore:
    def __init__(self, path, truncate=False):
        if truncate or not os.path.exists(path):
            f = open(path, "wb")
            f.truncate(MAX_FILESIZE)
            f.close()
        self._file = open(path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), 0)
        self._offset = 0

    def close(self):
        self._mm.flush()
        self._mm.close()
        self._file.close()

    def write(self, msg_id, body):
        if isinstance(body, str):
            body = body.encode()
        length = len(body)
        if self._offset + HEADER_SIZE + length > MAX_FILESIZE:
            print("no space left")
            return

        timestamp = int(time.time() * 1000)
        # write fixed header, then variable body
        self._mm.seek(self._offset)
        self._mm.write(struct.pack(HEADER_FORMAT, msg_id, length, timestamp))
        self._mm.write(body)
        self._offset += HEADER_SIZE + length

    def read(self):
        self._mm.seek(self._offset)
        buf = self._mm.read(HEADER_SIZE)
        if len(buf) < HEADER_SIZE:
            return None, None

        msg_id, length, timestamp = struct.unpack(HEADER_FORMAT, buf)
        if length == 0:
            return None, None

        body = self._mm.read(length)
        self._offset += HEADER_SIZE + length
        header = {"msg_id": msg_id, "length": length, "timestamp": timestamp}
        return header, body


# write
s = MsgStore(DB_FILE, truncate=True)
s.write(1, "hello")
s.write(2, "this is a longer message with more content")
s.write(3, "short")
s.write(4, b"X" * 500)
print(f"=== wrote 4 messages, total {s._offset} bytes used ===\n")
s.close()

# re-open and read
print("=== re-open and read ===")
s = MsgStore(DB_FILE)
while True:
    hdr, body = s.read()
    if hdr is None:
        break
    preview = body[:40].decode(errors="replace")
    suffix = "..." if hdr["length"] > 40 else ""
    print(f"msg_id={hdr['msg_id']:<3} len={hdr['length']:<4} ts={hdr['timestamp']} body=\"{preview}{suffix}\"")
s.close()

os.unlink(DB_FILE)

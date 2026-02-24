"""Python mmap comparison: same struct persistence, but requires manual packing."""
import mmap
import struct
import os

DB_FILE = "/tmp/mmap_users_py.db"
USER_FMT = "i32sf"  # int, 32-char string, float
USER_SIZE = struct.calcsize(USER_FMT)
HEADER_FMT = "i"  # count
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAX_USERS = 100
DB_SIZE = HEADER_SIZE + USER_SIZE * MAX_USERS

def db_open(path):
    """In Python, mmap gives you a bytearray-like object, not a struct pointer."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, DB_SIZE)
    mm = mmap.mmap(fd, DB_SIZE)
    os.close(fd)
    return mm

def db_add(mm, id, name, score):
    count = struct.unpack_from(HEADER_FMT, mm, 0)[0]
    # must manually pack the struct — no direct pointer access like C
    offset = HEADER_SIZE + count * USER_SIZE
    struct.pack_into(USER_FMT, mm, offset, id, name.encode().ljust(32, b'\x00'), score)
    struct.pack_into(HEADER_FMT, mm, 0, count + 1)

def db_print(mm):
    count = struct.unpack_from(HEADER_FMT, mm, 0)[0]
    print(f"UserDB: {count} users")
    for i in range(count):
        offset = HEADER_SIZE + i * USER_SIZE
        id, name, score = struct.unpack_from(USER_FMT, mm, offset)
        print(f"  [{i}] id={id} name={name.decode().strip(chr(0)):10s} score={score:.1f}")

# write
print(f"=== writing to {DB_FILE} ===")
mm = db_open(DB_FILE)
struct.pack_into(HEADER_FMT, mm, 0, 0)  # reset count
db_add(mm, 1, "Alice", 95.5)
db_add(mm, 2, "Bob", 87.3)
db_add(mm, 3, "Charlie", 91.0)
db_print(mm)
mm.flush()
mm.close()

# re-open: data persists
print("\n=== re-opening (data persisted) ===")
mm = db_open(DB_FILE)
db_print(mm)
mm.close()

# KEY DIFFERENCE: Python requires struct.pack/unpack for every read/write.
# In C, the struct IS the memory layout — zero overhead.
print(f"\nPython struct size per user: {USER_SIZE} bytes (vs C: same, but no pack/unpack cost)")

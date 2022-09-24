# https://docs.python.org/3/library/struct.html
import struct

format = {"a": "h", "b": "h", "c": "h"}
data = {"a": 1, "b": 2, "c": 3}
key = ["a", "b", "c"]

format_str = "".join(f for f in format.values())
values = [data[k] for k in key]

# encode to binary
encode = struct.pack(format_str, *values)
print(f"encode={encode}")

# decode from binary
decode = struct.unpack(format_str, encode)
print(f"decode={decode}")
import opendal

op = opendal.Operator("fs", root=".")
op.write("test.txt", b"Hello World")
print(op.read("test.txt"))
print(op.stat("test.txt").content_length)

#!/usr/bin/env python3
from bcc import BPF

b = BPF(src_file="hello_world.c")
b.attach_kprobe(event="do_sys_openat2", fn_name="hello_world")
b.trace_print()
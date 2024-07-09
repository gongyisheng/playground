#!/usr/bin/env python3
from bcc import BPF

b = BPF(src_file="monitor.c")
# do_sys_read
b.attach_kprobe(event="sys_read", fn_name="do_sys_read")
# do_sys_write
b.attach_kprobe(event="sys_write", fn_name="do_sys_write")
# do_tcp_v4_sendmsg
b.attach_kprobe(event="tcp_v4_sendmsg", fn_name="do_tcp_v4_sendmsg")
# do_tcp_v4_recvmsg
b.attach_kprobe(event="tcp_v4_recvmsg", fn_name="do_tcp_v4_recvmsg")
b.trace_print()
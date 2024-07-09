int do_sys_read(void *ctx)
{
    bpf_trace_printk("do_sys_read");
    return 0;
}

int do_sys_write(void *ctx)
{
    bpf_trace_printk("do_sys_write");
    return 0;
}

int do_tcp_v4_sendmsg(void *ctx)
{
    bpf_trace_printk("do_tcp_v4_sendmsg");
    return 0;
}

int do_tcp_v4_recvmsg(void *ctx)
{
    bpf_trace_printk("do_tcp_v4_recvmsg");
    return 0;
}


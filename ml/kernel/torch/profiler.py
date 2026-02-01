import torch

# ## Default way to use profiler
# with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     for _ in range(10):
#         a = torch.square(torch.randn(10000, 10000).cuda())

# prof.export_chrome_trace("trace.json")


## With warmup and skip
# https://pytorch.org/docs/stable/profiler.html

# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=-1))
    # Print device (GPU) time if available
    print("\n--- Device (GPU) Times ---")
    for event in prof.key_averages():
        if event.device_time_total > 0:
            print(f"{event.key}: GPU time = {event.device_time_total / 1000:.3f}ms")
    # Uncomment to save trace file
    # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")


# Warmup CUDA before profiling - run actual operations to initialize GPU
_ = torch.randn(1000, 1000, device='cuda')
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,

    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    on_trace_ready=trace_handler
    ) as p:
        for iter in range(10):
            x = torch.randn(10000, 10000, device='cuda')
            y = torch.square(x)
            torch.cuda.synchronize()  # Wait for GPU to finish - required for accurate timing
            p.step()


# Alternative: Manual CUDA event timing (works without CUPTI permissions)
# Use this if the profiler above shows 0 for GPU times
print("\n" + "="*60)
print("Alternative: CUDA Event Timing (no CUPTI needed)")
print("="*60)

# Warmup
for _ in range(3):
    _ = torch.square(torch.randn(10000, 10000, device='cuda'))
torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Time randn
start_event.record()
x = torch.randn(10000, 10000, device='cuda')
end_event.record()
torch.cuda.synchronize()
print(f"randn(10000,10000):  {start_event.elapsed_time(end_event):.3f} ms")

# Time square
start_event.record()
y = torch.square(x)
end_event.record()
torch.cuda.synchronize()
print(f"square:              {start_event.elapsed_time(end_event):.3f} ms")
import multiprocessing as mp
import time
import zmq

WORK_ADDR = "tcp://127.0.0.1:5557"
RESULT_ADDR = "tcp://127.0.0.1:5558"
N_WORKERS = 3


def worker(worker_id: int) -> None:
    ctx = zmq.Context()
    work = ctx.socket(zmq.PULL)
    work.setsockopt(zmq.LINGER, 0)
    work.connect(WORK_ADDR)
    results = ctx.socket(zmq.PUSH)
    results.setsockopt(zmq.LINGER, 0)
    results.connect(RESULT_ADDR)

    while True:
        msg = work.recv_pyobj()
        if msg is None:
            break
        idx, text = msg
        tokens = text.lower().split()
        results.send_pyobj((worker_id, idx, tokens))

    work.close()
    results.close()
    ctx.term()


def main() -> None:
    ctx = zmq.Context()
    work = ctx.socket(zmq.PUSH)
    work.setsockopt(zmq.LINGER, 0)
    work.bind(WORK_ADDR)
    results = ctx.socket(zmq.PULL)
    results.setsockopt(zmq.LINGER, 0)
    results.bind(RESULT_ADDR)

    spawn = mp.get_context("spawn")
    procs = [spawn.Process(target=worker, args=(i,)) for i in range(N_WORKERS)]
    for p in procs:
        p.start()

    time.sleep(0.5)

    texts = [
        "Hello world from ZMQ",
        "Distributed tokenization is fun",
        "Parallel pipeline pattern works well",
        "PUSH PULL load balances automatically",
        "Sentinel values terminate workers cleanly",
        "ZeroMQ has no broker by design",
    ]

    for i, text in enumerate(texts):
        work.send_pyobj((i, text))

    collected = {}
    for _ in range(len(texts)):
        worker_id, idx, tokens = results.recv_pyobj()
        collected[idx] = tokens
        print(f"worker {worker_id} -> idx={idx}: {tokens}")

    # gracefully exit
    for _ in range(N_WORKERS):
        work.send_pyobj(None)
    for p in procs:
        p.join()

    work.close()
    results.close()
    ctx.term()

    print("\nfinal ordered output:")
    for idx in sorted(collected):
        print(f"  {idx}: {collected[idx]}")


if __name__ == "__main__":
    main()

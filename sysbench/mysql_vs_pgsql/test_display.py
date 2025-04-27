results = {
    "oltp_point_select": {
        1: {
            "read": 1000,
            "write": 2000,
            "transaction": 3000,
        },
        2: {
            "read": 4000,
            "write": 5000,
            "transaction": 6000,
        },
        4: {
            "read": 7000,
            "write": 8000,
            "transaction": 9000,
        },
    },
    "oltp_update_non_index": {
        1: {
            "read": 1000,
            "write": 2000,
            "transaction": 3000,
        },
        2: {
            "read": 4000,
            "write": 5000,
            "transaction": 6000,
        },
        4: {
            "read": 7000,
            "write": 8000,
            "transaction": 90000,
        }
    }
}


def display_results(results):
    print("Results:")
    METHOD_LENGTH = max([len(method) for method in results.keys()] + [len("Method")])
    THREAD_LENGTH = max([len(str(thread)) for thread_results in results.values() for thread in thread_results.keys()] + [len("Threads")])
    READ_LENGTH = max([len(str(values['read'])) for thread_results in results.values() for values in thread_results.values()] + [len("Read QPS")])
    WRITE_LENGTH = max([len(str(values['write'])) for thread_results in results.values() for values in thread_results.values()] + [len("Write QPS")])
    TRANSACTION_LENGTH = max([len(str(values['transaction'])) for thread_results in results.values() for values in thread_results.values()] + [len("Transaction QPS")])
    first_line = f"| Method{' '*(METHOD_LENGTH - len('Method'))} | Threads{' '*(THREAD_LENGTH - len('Threads'))} | Read QPS{' '*(READ_LENGTH - len('Read QPS'))} | Write QPS{' '*(WRITE_LENGTH - len('Write QPS'))} | Transaction QPS{' '*(TRANSACTION_LENGTH - len('Transaction QPS'))} |"
    second_line = f"|{'-'*(METHOD_LENGTH + 2)}|{'-'*(THREAD_LENGTH + 2)}|{'-'*(READ_LENGTH + 2)}|{'-'*(WRITE_LENGTH + 2)}|{'-'*(TRANSACTION_LENGTH + 2)}|"
    print(first_line)
    print(second_line)
    for bench_name, thread_results in results.items():
        for thread, values in sorted(thread_results.items(), key=lambda x: x[0]):
            line = f"| {bench_name}{' '*(METHOD_LENGTH - len(bench_name))} | {thread}{' '*(THREAD_LENGTH - len(str(thread)))} | {values['read']}{' '*(READ_LENGTH - len(str(values['read'])))} | {values['write']}{' '*(WRITE_LENGTH - len(str(values['write'])))} | {values['transaction']}{' '*(TRANSACTION_LENGTH - len(str(values['transaction'])))} |"
            print(line)

display_results(results)
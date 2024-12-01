import re
import subprocess

def build_commands(user, password, database, table_size, bench_name, threads):
    prep_command = ["sysbench", "--db-driver=pgsql", "--pgsql-host=127.0.0.1", "--pgsql-port=5432", f"--pgsql-user={user}" f"--pgsql-password={password}" f"--pgsql-db={database}", bench_name, "prepare", f"--table-size={table_size}"]
    run_command = ["sysbench", "--db-driver=pgsql", "--pgsql-host=127.0.0.1", "--pgsql-port=5432", f"--pgsql-user={user}" f"--pgsql-password={password}" f"--pgsql-db={database}", bench_name, "run", f"--threads={threads}"]
    clean_command = ["sysbench", "--db-driver=pgsql", "--pgsql-host=127.0.0.1", "--pgsql-port=5432", f"--pgsql-user={user}" f"--pgsql-password={password}" f"--pgsql-db={database}", bench_name, "cleanup"]
    return prep_command, run_command, clean_command

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

def run_sysbench(user, password, database, table_size):
    bench_set = [
        "oltp_insert",
        "oltp_point_select",
        "select_random_points",
        "select_random_ranges",
        "oltp_read_only",
        "oltp_read_write",
        "oltp_write_only",
        "oltp_update_index",
        "oltp_update_non_index",
    ]

    # Regular expressions to extract data
    read_pattern = r'read:\s+(\d+)'
    write_pattern = r'write:\s+(\d+)'
    transaction_pattern = r'transactions:\s+(\d+)'
    time_pattern = r'total time:\s+([\d.]+)s'

    print(f"Running sysbench benchmarks: database={database}, table_size={table_size}")
    results = {}

    threads = [1, 2, 4, 8, 16, 32, 64]
    for bench_name in bench_set:
        for thread in threads:
            prep_command, run_command, clean_command = build_commands(user, password, database, table_size, bench_name, thread)
            print(f"Running {bench_name} with {thread} threads")
            subprocess.run(prep_command, capture_output=True, text=True)
            result = subprocess.run(run_command, capture_output=True, text=True)
            subprocess.run(clean_command, capture_output=True, text=True)
            output = result.stdout
            # Extracting the values
            read = int(re.search(read_pattern, output).group(1))
            write = int(re.search(write_pattern, output).group(1))
            transaction = float(re.search(transaction_pattern, output).group(1))
            time = float(re.search(time_pattern, output).group(1))
            # Display extracted values
            print("Read QPS:", read/time)
            print("Write QPS:", write/time)
            print("Transaction QPS:", transaction/time)
            # Save the extracted values
            if bench_name not in results:
                results[bench_name] = {}
            results[bench_name][thread] = {
                "read": read/time,
                "write": write/time,
                "transaction": transaction/time
            }
    display_results(results)


if __name__ == "__main__":
    import sys
    user = sys.argv[1]
    password = sys.argv[2]
    database = sys.argv[3] if len(sys.argv) > 3 else "sbtest"
    table_size = sys.argv[4] if len(sys.argv) > 4 else 1000000
    run_sysbench(user, password, database, table_size)
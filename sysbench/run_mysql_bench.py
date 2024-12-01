import re
import subprocess

def build_commands(user, password, database, bench_name, threads):
    prep_command = ["sysbench", "--db-driver=mysql", "--mysql-host=127.0.0.1", "--mysql-port=3306", f"--mysql-user={user}", f"--mysql-password={password}", f"--mysql-db={database}", bench_name, "prepare"]
    run_command = ["sysbench", "--db-driver=mysql", "--mysql-host=127.0.0.1", "--mysql-port=3306", f"--mysql-user={user}", f"--mysql-password={password}", f"--mysql-db={database}", bench_name, "run", f"--threads={threads}"]
    clean_command = ["sysbench", "--db-driver=mysql", "--mysql-host=127.0.0.1", "--mysql-port=3306", f"--mysql-user={user}", f"--mysql-password={password}", f"--mysql-db={database}", bench_name, "cleanup"]
    return prep_command, run_command, clean_command

def run_sysbench(user, password, database):
    read_bench_set = [
        "oltp_point_select",
        "oltp_read_only",
        "select_random_points",
        "select_random_ranges",
    ]
    write_bench_set = [
        "oltp_delete_insert",
        "oltp_insert",
        "oltp_read_write",
        "oltp_update_index",
        "oltp_update_non_index",
        "oltp_write_only",
        "types_delete_insert",
    ]

    # Regular expressions to extract data
    read_pattern = r'read:\s+(\d+)'
    write_pattern = r'write:\s+(\d+)'
    transaction_pattern = r'transactions:\s+(\d+)'
    time_pattern = r'total time:\s+([\d.]+)s'

    threads = [1, 2, 4, 8, 16, 32, 64]
    for bench_name in read_bench_set:
        for thread in threads:
            prep_command, run_command, clean_command = build_commands(user, password, database, bench_name, thread)
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
            break
        break

if __name__ == "__main__":
    import sys
    user = sys.argv[1]
    password = sys.argv[2]
    database = sys.argv[3] if len(sys.argv) > 3 else "sbtest"
    run_sysbench(user, password, database)
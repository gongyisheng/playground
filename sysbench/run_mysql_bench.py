import subprocess

def build_commands(user, password, database, bench_name, threads):
    prep_command = ["sysbench", "--db-driver=mysql", "--mysql-host=127.0.0.1", "--mysql-port=3306", f"--mysql-user={user}", f"--mysql-password={password}", f"--mysql-db={database}", bench_name, "prepare"]
    run_command = ["sysbench", "--db-driver=mysql", "--mysql-host=127.0.0.1", "--mysql-port=3306", f"--mysql-user={user}", f"--mysql-password={password}", f"--mysql-db={database}", bench_name, "run", f"--threads={threads}"]
    clean_command = ["sysbench", "--db-driver=mysql", "--mysql-host=127.0.0.1", "--mysql-port=3306", f"--mysql-user={user}", f"--mysql-password={password}", f"--mysql-db={database}", bench_name, "cleanup"]
    return prep_command, run_command, clean_command

def run_sysbench(user, password, database):
    read_bench_set = [
        "covering_index_scan",
        "index_join",
        "index_join_scan",
        "oltp_point_select",
        "oltp_read_only",
        "select_random_points",
        "select_random_ranges",
        "index_scan",
        "groupby_scan",
        "table_scan",
        "types_table_scan"
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

    threads = [1, 2, 4, 8, 16, 32, 64]
    for bench_name in read_bench_set:
        for thread in threads:
            prep_command, run_command, clean_command = build_commands(user, password, database, bench_name, thread)
            print(f"Running {bench_name} with {thread} threads")
            subprocess.run(prep_command, capture_output=True, text=True)
            result = subprocess.run(run_command, capture_output=True, text=True)
            subprocess.run(clean_command, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stderr)
                print(result.stdout)
                break
            else:
                print(result.stderr)
                print(result.stdout)
            break
        break

if __name__ == "__main__":
    import sys
    user = sys.argv[1]
    password = sys.argv[2]
    database = sys.argv[3] if len(sys.argv) > 3 else "sbtest"
    run_sysbench(user, password, database)
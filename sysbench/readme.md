# install
```
sudo apt install sysbench
man sysbench
```

# cpu benchmark
```
sysbench cpu run
```

benchmark config:
```
sysbench 1.0.20
Running the test with following options:
Number of threads: 1
Prime numbers limit: 10000
```

benchmark result:
| Model            | CPU core         | Events per sec |
|------------------|------------------|----------------|
| Raspberrypi 5b   | ARM Cortex-A76   | 2726.31        |
| AWS r7g.2xlarge  | Amazon Graviton3 | 3024.99        |
| Macbook pro 2020 | Apple M1         | 12310876.76    |

# memory benchmark
```
sysbench memory run
```

benchmark config:
```
sysbench 1.0.20
Running memory speed test with the following options:
  block size: 1KiB
  total size: 102400MiB
  operation: write
  scope: global
```

benchmark result:
| Model            | Memory type        | Transfer speed MiB/s |
|------------------|--------------------|----------------------|
| Raspberrypi 5b   | LPDDR4X            | 3636.61              |
| AWS r7g.2xlarge  | DDR5-5600          | 4777.81              |
| Macbook pro 2020 | LPDDR4X            | 8609.73              |

# diskio benchmark
```
sysbench fileio --file-total-size=5G --file-num=5 prepare

# sequential access
sysbench fileio --file-total-size=5G --file-num=5 --file-io-mode=async --file-fsync-freq=0 --file-test-mode=seqrd --file-block-size=1M run
sysbench fileio --file-total-size=5G --file-num=5 --file-io-mode=async --file-fsync-freq=0 --file-test-mode=seqwr --file-block-size=1M run

# random access
sysbench fileio --file-total-size=5G --file-num=5 --file-io-mode=async --file-fsync-freq=0 --file-test-mode=rndrd --file-block-size=4k run
sysbench fileio --file-total-size=5G --file-num=5 --file-io-mode=async --file-fsync-freq=0 --file-test-mode=rndwr --file-block-size=4k run
```

benchmark config:
```
sysbench 1.0.20
Running the test with following options:
Number of threads: 1
Extra file open flags: (none)
5 files, 1GiB each
5GiB total file size
Block size 1MiB
Calling fsync() at the end of test, Enabled.
Using asynchronous I/O mode
```

benchmark result:
| Model            | Disk type          | seqrd speed MiB/s | seqwr speed MiB/s | rndrd speed MiB/s | rndwr speed MiB/s |
|------------------|--------------------|-------------------|-------------------|-------------------|-------------------|
| Raspberrypi 5b   | TF card            | 424.69            | 27.99             | 104.41            | 3.48              |
| Raspberrypi 5b   | HDD                | 90.58             | 94.80             | 0.74              | 1.70              |
| AWS r7g.2xlarge  | NVMe               | 7882.92           | 1095.23           | 1954.06           | 376.19            | 
| Macbook pro 2020 | SSD                | 1988.67           | 972.30            | 45.86             | 36.03             |

# mysql benchmark
```
# 1. create a database for benchmark
CREATE DATABASE sbtest;

# 2. run sysbench
sysbench --db-driver=mysql --mysql-host=127.0.0.1 --mysql-port=3306 --mysql-user=root --mysql-password=<password> --mysql-db=sbtest oltp_read_write prepare
sysbench --db-driver=mysql --mysql-host=127.0.0.1 --mysql-port=3306 --mysql-user=root --mysql-password=<password> --mysql-db=sbtest oltp_read_write run --threads=8
sysbench --db-driver=mysql --mysql-host=127.0.0.1 --mysql-port=3306 --mysql-user=username --mysql-password=<password> --mysql-db=sbtest oltp_read_write cleanup

# Note: test list
oltp_insert
oltp_point_select
oltp_read_only
oltp_read_write
oltp_write_only
oltp_update_index
oltp_update_non_index
```

benchmark config:
```
on raspberrypi 5b
sysbench 1.0.20
Number of threads: 8
```

benchmark result:
| Method                | Read QPS | Write QPS | Transaction QPS |
|-----------------------|----------|-----------|-----------------|
| oltp_insert           | 0        | 162.71    | 162.71          |
| oltp_point_select     | 12615.31 | 0         | 12615.31        |
| select_random_points  | 7712.25  | 0         | 7712.25         |
| select_random_ranges  | 7339.17  | 0         | 7339.17         |
| oltp_read_only        | 8046.43  | 0         | 574.66          |
| oltp_read_write       | 1432.49  | 409.28    | 102.30          |
| oltp_write_only       | 0        | 491.87    | 122.79          |
| oltp_update_index     | 0        | 151.30    | 151.30          |
| oltp_update_non_index | 0        | 129.35    | 129.35          |

# postgre benchmark
```
sysbench --db-driver=pgsql --pgsql-host=127.0.0.1 --pgsql-port=5432 --pgsql-user=root --pgsql-password=<password> --pgsql-db=sbtest oltp_insert prepare
sysbench --db-driver=pgsql --pgsql-host=127.0.0.1 --pgsql-port=5432 --pgsql-user=root --pgsql-password=<password> --pgsql-db=sbtest oltp_insert run --threads=8
sysbench --db-driver=pgsql --pgsql-host=127.0.0.1 --pgsql-port=5432 --pgsql-user=root --pgsql-password=<password> --pgsql-db=sbtest oltp_insert cleanup
```

benchmark config:
```
sysbench 1.0.20
```

benchmark result:
| Method                | Read QPS | Write QPS | Transaction QPS |
|-----------------------|----------|-----------|-----------------|
| oltp_insert           | 0        | 1839.41   | 1839.41         |
| oltp_point_select     | 22147.53 | 0         | 22147.53        |
| select_random_points  | 17985.42 | 0         | 17985.42        |
| select_random_ranges  | 11068.39 | 0         | 11068.39        |
| oltp_read_only        | 14891.86 | 0         | 1063.55         |
| oltp_read_write       | 9031.64  | 2047.06   | 470.79          |
| oltp_write_only       | 0        | 2175.91   | 501.48          |
| oltp_update_index     | 0        | 1927.78   | 1927.78         |
| oltp_update_non_index | 0        | 1831.24   | 1831.24         |

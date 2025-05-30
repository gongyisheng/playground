# rw splitting 
## test commands
```
# prepare
sysbench \
  --db-driver=mysql \
  --mysql-host=cluster-n0.local \
  --mysql-user=dev \
  --mysql-password=dev \
  --mysql-db=rw_splitting_test \
  --threads=128 \
  --time=0 \
  --events=1000000 \
  prepare.lua run

# run select_80_update_20
sysbench \
  --db-driver=mysql \
  --mysql-host=cluster-n0.local \
  --mysql-user=dev \
  --mysql-password=dev \
  --mysql-db=rw_splitting_test \
  --threads=128 \
  --time=0 \
  --events=1000000 \
  prepare.lua run
```

## result
```
# hardware
raspberrypi 5 8GB
SK Hynix PC601 1TB (PCIe 3.0)

# hardware benchmark
nvme max throughput: 430 MiB/s
nvme max iops: 30K io/s
network max throughput: 1 Gib/s (125 MiB/s)

prepare stage: 
128 thread reach max
write QPS = 4000
iops = 6000
write throughput = 150MB/s
```

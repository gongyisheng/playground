# rw splitting 
## test commands
```
# prepare
## local
sysbench \
  --db-driver=mysql \
  --mysql-host=127.0.0.1 \
  --mysql-user=dev \
  --mysql-password=dev \
  --mysql-db=rw_splitting_test \
  --threads=128 \
  --time=0 \
  --events=1000000 \
  prepare.lua run

## remote
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

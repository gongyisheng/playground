# rw splitting 
## test commands
```
# prepare
sysbench \
  --db-driver=mysql \
  --mysql-host=127.0.0.1 \
  --mysql-user=dev \
  --mysql-password=dev \
  --mysql-db=rw_splitting_test \
  --threads=8 \
  --time=60 \
  prepare.lua run
```

## result

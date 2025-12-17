# miles

## install
```
sudo docker pull radixark/miles:latest
sudo docker create --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 -v .:/root/workspace --name miles radixark/miles:latest sleep infinity
sudo docker start miles
sudo docker exec -it miles bash
```
# docker case
## question
Make sure 2 containers can connect to each other

## method 1
use docker command line
```
sudo docker network create my_network

sudo docker run -d --name c1 --network my_network nginx
sudo docker run -d --name c2 --network my_network nginx

docker exec -it container1 ping container2
```

## method 2

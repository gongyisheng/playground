es  
`docker pull elasticsearch:8.3.1`  
`docker network create es_net`
`docker run -d --name elasticsearch --net es_net -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:8.3.1`

kibana  
`docker pull kibana:8.3.2`  
`docker run -d --name kibana --net es_net -p 5601:5601 kibana:8.3.2`  
go to `http://127.0.0.1:5601/`

create kibana enrollment token  
es: `bin/elasticsearch-create-enrollment-token --scope kibana`  
kibana: `bin/kibana-verification-code`

create user  
`bin/elasticsearch-users useradd <user_name> -p <pwd> -r kibana_admin`

ref:  
`https://medium.com/@teeppiphat/install-elasticsearch-docker-on-macos-m1-7dfbb8876b99`

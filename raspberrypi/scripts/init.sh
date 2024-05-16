is_master=$1

echo "Is master: $is_master"

cd ~
git config --global user.name "gongyisheng"
git config --global user.email "yisheng_gong@onmail.com"
git config --global credential.helper store

git clone https://github.com/gongyisheng/user-key.git

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install vim tcpdump linux-tools-common linux-tools-generic dnsutils net-tools wireless-tools build-essential -y

# 1. Install docker
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable docker.service
sudo systemctl enable containerd.service

# 2. Install node-exporter
mkdir -p ~/monitoring/node-exporter
cd ~/monitoring/node-exporter
sudo docker run -d --name="node-exporter" --net="host" --pid="host" -v "/:/host:ro,rslave" --restart=always quay.io/prometheus/node-exporter:latest --path.rootfs=/host

# 3. Install prometheus (only on master)
mkdir -p ~/monitoring/prometheus
cd ~/monitoring/prometheus
cp ~/user-key/prometheus/prometheus.yml .
sudo docker run -d --name prometheus --net="host" --pid="host" -p 9090:9090 -v ~/monitoring/prometheus:/etc/prometheus --restart always prom/prometheus

# 4. Install grafana (only on master)
mkdir -p ~/monitoring/grafana
cd ~/monitoring/grafana
sudo docker run -d --name=grafana --net="host" --pid="host" -p 3000:3000 --restart always grafana/grafana

# 5. Install cloudflare tunnel
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
sudo cloudflared service install XXXXX
sudo systemctl enable cloudflared

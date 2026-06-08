# Last updated: May 14, 2026
# running on x86_64, ubuntu 24.04

cd ~
git config --global user.name "gongyisheng"
git config --global user.email "yisheng_gong@onmail.com"
git config --global credential.helper store

sudo apt-get update
sudo apt-get install curl vim tcpdump linux-tools-common linux-tools-generic dnsutils net-tools wireless-tools build-essential python3-pip -y

# 1. Install cloudflare tunnel
# ref: cloudflare -> zero trust -> network -> connectors
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-public-v2.gpg | sudo tee /usr/share/keyrings/cloudflare-public-v2.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-public-v2.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared
sudo cloudflared service install "API_KEY"

# 2. Install docker
# ref: https://docs.docker.com/engine/install/ubuntu/
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 3. nvidia related
# install cuda and driver
# ref: https://developer.nvidia.com/cuda-13-0-0-download-archive
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

sudo apt-get install -y nvidia-open

# install container toolkit
# ref: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo apt-get update && sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}


# 4. clone repos
git clone https://github.com/gongyisheng/xxxx.git
cd xxxx
cd node_exporter && sudo docker compose up -d
cd ../prometheus && sudo docker compose up -d
cd ../nvidia_gpu_exporter && sudo docker compose up -d

# 5. env softwares
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
# claude
curl -fsSL https://claude.ai/install.sh | bash
echo 'alias yolo="claude --dangerously-skip-permissions"' >> ~/.bashrc && source ~/.bashrc
mkdir -p ~/.claude && curl -fsSL https://raw.githubusercontent.com/gongyisheng/playground/dev/ai_tools/claude_code/CLAUDE.md -o ~/.claude/CLAUDE.md
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~

# node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 24
node -v # Should print "v24.15.0".
npm -v # Should print "11.12.1".
# ccusage
npm install -g ccusage

# last: upgrade all apt dependencies
sudo apt-get update
sudo apt-get upgrade

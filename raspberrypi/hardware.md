# cpu info
`lscpu`  
`cat /proc/cpuinfo`  

# memory info
`free -m`  
`free -h`  

# setup swap
change default system-level swap size (only for raspberry pi os)  
`sudo vim /etc/dphys-swapfile`  
`CONF_SWAPFILE=4096`(MB)  
`CONF_MAXSWAP=8192`(MB)   
flush existing swap  
`sudo swapoff -a`  
add new swap files (bs=size per file, count=file count, bs*count=swap size)  
`sudo dd if=/dev/zero of=/var/swap bs=4M count=1024`  
initalize(format) swap files  
`sudo mkswap /var/swap`  
get partition uuid   
`sudo blkid /var/swap`  
turn on swap  
`sudo swapon /var/swap` or `sudo swapon -U <UUID>`  
edit `/etc/fstab` file, start swap after boot  
`UUID=<UUID> none swap sw 0 0`  
reboot
`sudo reboot`

# disk info
`df -h`

# disk health check
`sudo badblocks -s -v /dev/sda`

# disk benchmark
use fio as the benchmark tool
```
# io_depth: I/O depth refers to the number of I/O operations that can be in progress at the same time for each worker thread

# install fio
sudo apt install -y fio

# create folder
TEST_DIR=./fiotest
sudo mkdir -p $TEST_DIR

# write throughput: sequential writes, 10G*16, block_size=1M, io_depth=64
sudo fio --name=write_throughput --directory=$TEST_DIR --numjobs=16 \
--size=10G --time_based --runtime=5m --ramp_time=2s --ioengine=libaio \
--direct=1 --verify=0 --bs=1M --iodepth=64 --rw=write \
--group_reporting=1 --iodepth_batch_submit=64 \
--iodepth_batch_complete_max=64

# write iops: random writes, 10G*1, block_size=4kb, io_depth=256
 sudo fio --name=write_iops --directory=$TEST_DIR --size=10G \
--time_based --runtime=5m --ramp_time=2s --ioengine=libaio --direct=1 \
--verify=0 --bs=4K --iodepth=256 --rw=randwrite --group_reporting=1  \
--iodepth_batch_submit=256  --iodepth_batch_complete_max=256

# read throughput: sequential reads, 10G*16, block_size=1M, io_depth=64
sudo fio --name=read_throughput --directory=$TEST_DIR --numjobs=16 \
--size=10G --time_based --runtime=5m --ramp_time=2s --ioengine=libaio \
--direct=1 --verify=0 --bs=1M --iodepth=64 --rw=read \
--group_reporting=1 \
--iodepth_batch_submit=64 --iodepth_batch_complete_max=64

# read iops: random reads, 10G*1, block_size=4kb, io_depth=256
sudo fio --name=read_iops --directory=$TEST_DIR --size=10G \
--time_based --runtime=5m --ramp_time=2s --ioengine=libaio --direct=1 \
--verify=0 --bs=4K --iodepth=256 --rw=randread --group_reporting=1 \
--iodepth_batch_submit=256  --iodepth_batch_complete_max=256

# clean up
sudo rm $TEST_DIR/write* $TEST_DIR/read*
```
reference: https://cloud.google.com/compute/docs/disks/benchmarking-pd-performance-linux

# stress test
`sudo apt-get install stress`
`while true; do vcgencmd measure_clock arm; vcgencmd measure_temp; sleep 10; done& stress -c 4 -t 900s`

# external disk mount
`sudo fdisk -l` see drivers number  
`sudo blkid` get UUID of a device
`sudo mount /dev/sda# /media/usbdisk` mount disk  
add following line to `/etc/fstab` to mount disk after boot  
`UUID=XXXXXXX /media/usbdisk auto nosuid,nodev,nofail 0 2`
unmount: IMPORTANT! dont use lazy umount if you cares driver can safely unplugged

# install nvme ssd
```
# Background
This project aims at using m2.nvme ssd as boot disk instead of USB/SD card.

USB has availability issues, it's not quite stable under moving/voltage change. I have experience that a usb-connected m2.nvme ssd dropped when a 2.5' HDD is plugged in through USB. The power supply does not have enough power to handle multiple external devices.

SD card is terrible at random read and write. It's not suitable for a desktop computer's boot disk.

# Method
material: raspberrypi 5, m.2 hat, m.2 nvme ssd

# 1. prepare image
# change boot order
sudo rpi-eeprom-config --edit
chage following:
BOOT_ORDER=0xf416
PCIE_PROBE=1

then reboot after these

after reboot, use sd card copier to copy image to nvme ssd

# 2. change config.txt
mount boot partition, edit config.txt
dtparam=pciex1
dtparam=pciex1_gen=3 # if you want to use gen 3.0, not recommend

# 3. remove sd card, install m2.nvme ssd/m2.pcie hat, reboot 
```
ref: https://wiki.geekworm.com/NVMe_SSD_boot_with_the_Raspberry_Pi_5

# wlan quality
`iwconfig wlan0 | grep Quality`   
`cat /proc/net/wireless` 
IMPORTANT: don't put HDD or metal thing too close to the board otherwise it affects wifi quality 

# change wifi
`sudo vim /etc/netplan/50-cloud-init.yaml`

```
# this file generated from information provided by the datasource. Changes
# to it will not persist across an instance reboot. To disable cloud-init’s
# network configuration capabilities, write a file 
# /etc/cloud/cloud.cfg.d/99-disable-network-config-cfg with the following:
# network:{config: disabled}
network:
    ethernets:
        etho:
            dhcp4: true
            optional: true
    version: 2
    wifis:
        wlan0:
            access-points:
                WIFI-NAME:
                    password: WIFI-PASSWORD
            dhcp4: true
            optional: true
```
ref: https://raspberrypi.stackexchange.com/questions/111722/rpi-4-running-ubuntu-server-20-04-cant-connect-to-wifi

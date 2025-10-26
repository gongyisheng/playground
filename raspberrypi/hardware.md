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

# turn off swap
`sudo dphys-swapfile swapoff` on raspberrypi os
`sudo dphys-swapfile uninstall` on raspberrypi os, disable auto start after reboot
`sudo update-rc.d dphys-swapfile remove` on raspberrypi os, disable upgrade
`sudo systemctl disable swapfile.swap` on ubuntu

# set swappiness
set up swap but only use when needed
```
echo "vm.swappiness=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

# disk info
`df -h`

# disk health check
`sudo badblocks -s -v /dev/sda`

# format a disk
`sudo umount /dev/sdX`  
`sudo mkfs.ext4 /dev/sda`  

# disk iostat
```
iostat -d -x -m 1 30

focus on:
%util: io utility, limit by device
await: read/write operation latency
aqu-sz: io queue size, determine whether it's blocked
areq-sz: average read/write request size, whether it's random I/O

other:
svctm: average service time (excluding queue time)
rqm/s: combined request number per sec
%rqm: rqm / (rw/s + rqm) * 100, combine efficiency (HDD: 70%-95%, SSD: 50%-80, NVMe: 0-30%)

cases:
1. high %util
for HDD, >80% is a warning
for SSD/NVMe, even ~100% is not an issue, focus on await and aqu-sz

2. high await
may caused by:
- high backend latency (eg, using microSD, random I/O, RAID, write amplification)
- frequently flush log

3. high %util + high await + high aqu-sz
device is the bottleneck
consider using RAID, or change I/O scheduler (SSD/NVMe: noop/none, HDD: deadline/bfq)

4. small areq-sz
small read and write I/O operations
considering use O_DIRECT

5. low util%, high await
invisible bottlenecks like
- high backend latency (using microSD, RAID, virtual disk like NAS)
- single thread block I/O
- wrong storage configuration (cross NUMA)
```

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

# suspend, hibernate, power off
```
suspend(sleep): need electricity, session keep in ram, low but not zero power usage
hibernate: don't need electricity, session keep from ram to swap, no power usage
power off: don't need electricity, session data lost
```
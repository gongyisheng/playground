# Background
This project aims at using m2.nvme ssd as boot disk instead of USB/SD card.

USB has availability issues, it's not quite stable under moving/voltage change. I have experience that a usb-connected m2.nvme ssd dropped when a 2.5' HDD is plugged in through USB. The power supply does not have enough power to handle multiple external devices.

SD card is terrible at random read and write. It's not suitable for a desktop computer's boot disk.

# Method
material: raspberrypi 5, m.2 hat, m.2 nvme ssd
```
# 1. prepare image
```
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


# Maintenance tips for Raspberry Pi
## Software
1. **Update and Upgrade**:
    - Update the package list and upgrade the packages to the latest version.
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```
2. **Clean up**:
    - Remove the packages that are no longer required.
    ```bash
    sudo apt-get autoremove
    ```
## Hardware
1. **Disk Choice**
   - SD card for booting and system software. Backup frequently and assume all the data on the SD card will be lost.
   - SSD for data storage with hot data have read/write performance requirements. Backup regularly and assume all the data on the SSD will be lost after years of use.
   - HDD for data storage with cold data which may not be touched frequently. Backup regularly and assume all the data on the HDD will be lost after years of use.
2. **Cooling**
   - Install a cooling fan and heat sink to prevent the Raspberry Pi from overheating.
   - The temperature of the Raspberry Pi should be kept below 80°C. Too much heat may cause SD card corruption.
3. **Power Supply**
   - The Raspberry Pi 5 requires a 5V 3A power supply.
   - Assume there'll be power outages. In practice, every 2-3 weeks there will be a power outage, caused by human error, weather, or other reasons. Make sure the Raspberry Pi can boot up automatically after a power outage.
4. **Network**
   - If the Raspberry Pi is using a Wi-Fi connection, please use 5GHz Wi-Fi to get a better network connection.
   - Use a static IP address for the Raspberry Pi if a DNS server is running on it. 
5. **Backup**
   - Backup the system and data regularly.
   - S3 deep archive is a good choice of backup cloud storage.

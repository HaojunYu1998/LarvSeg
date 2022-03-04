#START

# Configure the Microsoft package repository
wget https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update

# Install blobfuse
sudo apt-get install blobfuse

# Optional
sudo mkdir /mnt/ramdisk_teamdrive
sudo mount -t tmpfs -o size=32g tmpfs /mnt/ramdisk_teamdrive
sudo mkdir /mnt/ramdisk_teamdrive/blobfusetmp
sudo chown yuhui /mnt/ramdisk_teamdrive/blobfusetmp # your user is the current username in the system.


# Configure your storage account credentials
# vim ~/fuse_connection.cfg
accountName msravcshare
accountKey 7DRoCM8UEzJzsRnrTRgUwUdO64gn9RHZurnw1PxWDAsTzGaP5BmLfvpDT1xeqAOw3gnIghsvMli7ekkYT6sZlw==
containerName teamdrive

# 
chmod 600 fuse_connection.cfg

# creat the temp folder to mount the blob
mkdir ~/mycontainer

# use the /mnt/ramdisk and add the 
sudo blobfuse ~/teamdrive --tmp-path=/mnt/ramdisk_teamdrive/blobfusetmp  --config-file=./fuse_connection_teamdrive.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other


sudo blobfuse ./teamdrive --tmp-path=/mnt/ramdisk_teamdrive/blobfusetmp  --config-file=./fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other


sudo blobfuse ./openseg_blob --tmp-path=/mnt/ramdisk_teamdrive/blobfusetmp  --config-file=./openseg_fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other


#DONE
# A2:D3:7A:FF:71:CF
# 54:E1:AD:F9:D0:78


# create ssh keys

ssh-keygen \
    -m PEM \
    -t rsa \
    -b 4096 \
    -C "yhyuan@pku.edu.cn" \
    -f ~/.ssh/rainbowsecret_privatekey \
    # -N rainbowsecret@msra

# setup env
sudo apt update
sudo apt-get install unzip -y
sudo apt-get install htop -y
sudo apt-get install tmux -y
sudo apt-get install vim -y

# sudo /opt/conda/bin/pip install protobuf==3.20.1
# sudo /opt/conda/bin/pip install wandb==0.11.0
sudo /opt/conda/bin/pip install timm
sudo mkdir -p /workspace
sudo ln -s /zeliuwestus2/dataset /workspace/dataset
cd /workspace/dataset/
sudo ln -s ./ade20k_full A847
sudo mkdir -p ./A847/images
sudo mkdir -p ./A847/annotations
sudo ln -s ./A847/train/image ./A847/images/training
sudo ln -s ./A847/train/label ./A847/annotations/training
sudo ln -s ./A847/val/image ./A847/images/validation
sudo ln -s ./A847/val/label ./A847/annotations/validation
sudo ln -s ./ADEChallengeData2016 A150
sudo ln -s ./imagenet22k_azcopy I21K
sudo ln -s ./coco_stuff164k C171
sudo ln -s ./C171/images/train2017 ./C171/images/training
sudo ln -s ./C171/images/val2017 ./C171/images/validation
sudo ln -s ./C171/annotations/train2017 ./C171/annotations/training
sudo ln -s ./C171/annotations/val2017 ./C171/annotations/validation


# /mnt/haojun/itpsea4data/azcopy_linux_amd64_10.15.0/azcopy copy "https://itpsea4data.blob.core.windows.net/v-miazhang/dataset/imagenet22k_azcopy/in21k_inter_ade_all.txt?sv=2021-04-10&st=2022-09-24T10%3A41%3A42Z&se=2022-09-30T10%3A41%3A00Z&sr=c&sp=racwl&sig=vz1rzj65A5OslGMxUVQdkVGjGLAnFwyUldJNRaryejc%3D" "https://zeliuwestus2.blob.core.windows.net/v-miazhang/dataset/imagenet22k_azcopy/?sv=2021-04-10&st=2022-09-24T10%3A41%3A07Z&se=2022-09-30T10%3A41%3A00Z&sr=c&sp=racwl&sig=peH2NM8T%2BaUAjPXb0%2BDz2hLxzato8pp%2Fz18oC56PyDE%3D" --recursive;

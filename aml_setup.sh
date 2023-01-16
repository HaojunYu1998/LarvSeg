# setup env
sudo apt update
sudo apt-get install unzip -y
sudo apt-get install htop -y
sudo apt-get install tmux -y
sudo apt-get install vim -y

sudo set -x
sudo set -e
# sudo /opt/conda/bin/pip install protobuf==3.20.1
# sudo /opt/conda/bin/pip install wandb==0.11.0
sudo /opt/conda/bin/pip install timm
sudo mkdir -p /workspace
sudo ln -s /zeliuwestus2/dataset /workspace/dataset
# A847
# sudo ln -s /zeliuwestus2/dataset/ade20k_full /mnt/haojun/dataset/A847

# sudo ln -s /zeliuwestus2/dataset/A847/train/image /mnt/haojun/dataset/A847/images/training
# sudo ln -s /zeliuwestus2/dataset/A847/train/label /mnt/haojun/dataset/A847/annotations/training
# sudo ln -s /zeliuwestus2/dataset/A847/val/image /mnt/haojun/dataset/A847/images/validation
# sudo ln -s /zeliuwestus2/dataset/A847/val/label /mnt/haojun/dataset/A847/annotations/validation
# # A150
# sudo ln -s /zeliuwestus2/dataset/ADEChallengeData2016 /mnt/haojun/dataset/A150
# # I21K
# sudo ln -s /zeliuwestus2/dataset/imagenet22k_azcopy /mnt/haojun/dataset/I21K
# # C171
# sudo ln -s /zeliuwestus2/dataset/coco_stuff164k /mnt/haojun/dataset/C171
# sudo ln -s /zeliuwestus2/dataset/C171/images/train2017 /mnt/haojun/dataset/C171/images/training
# sudo ln -s /zeliuwestus2/dataset/C171/images/val2017 /mnt/haojun/dataset/C171/images/validation
# sudo ln -s /zeliuwestus2/dataset/C171/annotations/train2017 /mnt/haojun/dataset/C171/annotations/training
# sudo ln -s /zeliuwestus2/dataset/C171/annotations/val2017 /mnt/haojun/dataset/C171/annotations/validation


# /mnt/haojun/itpsea4data/azcopy_linux_amd64_10.15.0/azcopy copy "https://itpsea4data.blob.core.windows.net/v-miazhang/dataset/imagenet22k_azcopy/in21k_inter_ade_all.txt?sv=2021-04-10&st=2022-09-24T10%3A41%3A42Z&se=2022-09-30T10%3A41%3A00Z&sr=c&sp=racwl&sig=vz1rzj65A5OslGMxUVQdkVGjGLAnFwyUldJNRaryejc%3D" "https://zeliuwestus2.blob.core.windows.net/v-miazhang/dataset/imagenet22k_azcopy/?sv=2021-04-10&st=2022-09-24T10%3A41%3A07Z&se=2022-09-30T10%3A41%3A00Z&sr=c&sp=racwl&sig=peH2NM8T%2BaUAjPXb0%2BDz2hLxzato8pp%2Fz18oC56PyDE%3D" --recursive;

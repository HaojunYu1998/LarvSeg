# setup env
sudo apt update
sudo apt-get install unzip -y
sudo apt-get install htop -y
sudo apt-get install tmux -y
sudo apt-get install vim -y
sudo /opt/conda/bin/pip install install git+https://github.com/lucasb-eyer/pydensecrf.git
cd third_party/CLIP
sudo /opt/conda/bin/pip install -e .
cd ../detectron2
sudo /opt/conda/bin/pip install -e .
cd ../..
sudo mkdir -p /mnt/haojun2
sudo ln -s /zeliuwestus2/dataset /mnt/haojun2/dataset

# sudo ln -s /zeliuwestus2/dataset /mnt/haojun2/dataset
# sudo ln -s /zeliuwestus2/dataset/ADEChallengeData2016 /mnt/haojun2/dataset/ADEChallengeData2016
# sudo ln -s /zeliuwestus2/dataset/ADE20K_2021_17_01 /mnt/haojun2/dataset/ADE20K_2021_17_01 
# sudo ln -s /zeliuwestus2/dataset/coco_stuff164k /mnt/haojun2/dataset/coco_stuff164k 
# sudo ln -s /zeliuwestus2/dataset/imagenet22k_azcopy /mnt/haojun2/dataset/imagenet22k_azcopy 

# /mnt/haojun/itpsea4data/azcopy_linux_amd64_10.15.0/azcopy copy "https://itpsea4data.blob.core.windows.net/v-miazhang/dataset/imagenet22k_azcopy?sv=2021-04-10&st=2022-09-24T10%3A41%3A42Z&se=2022-09-30T10%3A41%3A00Z&sr=c&sp=racwl&sig=vz1rzj65A5OslGMxUVQdkVGjGLAnFwyUldJNRaryejc%3D" "https://zeliuwestus2.blob.core.windows.net/v-miazhang/dataset/?sv=2021-04-10&st=2022-09-24T10%3A41%3A07Z&se=2022-09-30T10%3A41%3A00Z&sr=c&sp=racwl&sig=peH2NM8T%2BaUAjPXb0%2BDz2hLxzato8pp%2Fz18oC56PyDE%3D" --recursive;

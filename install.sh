pip install mmcv-full==1.3.12 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0a0+2ecb2c7/index.html
pip install -e .
cd third_party/CLIP
pip install -e .
cd ../../../detectron2
pip install -e .
cd ../mmseg
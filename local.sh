# #!/bin/bash


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

IMAGE="hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel"

TRY_CONFIG=$1
if [[ "$TRY_CONFIG"x =~ ^configs/.* ]]; then
    TASK_IS_EXPERIMENT=true
else
    TASK_IS_EXPERIMENT=false
fi

ARGS=("$@")
QUOTED_ARGS=$(printf "'%s' " "${ARGS[@]}")

read -d '' CMD <<EOF
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
ln -s /mdata/ade /data/ade
ln -s /mdata/ade20k_full /data/ade20k_full
ln -s /mdata/coco_stuff164k /data/coco_stuff164k
ln -s /mdata/imagenet21k /data/imagenet21k

pip install git+https://github.com/lucasb-eyer/pydensecrf.git
cd third_party/CLIP
pip install -e .
cd ../detectron2
pip install -e .

cd /workspace
if $TASK_IS_EXPERIMENT; then
    bash tools/dist_train.sh $QUOTED_ARGS
else
    $QUOTED_ARGS
fi
EOF

if [ $# == 0 ]; then
    CMD=bash
fi

if [ -t 1 ]; then
    FLAG="-it"
fi

echo "=========== COMMAND ==========="
echo "$CMD"
echo "==============================="

sudo nvidia-docker run \
    --rm --ipc=host ${FLAG} \
    -v "$PWD":/workspace \
    -v ~/itpsea4data/dataset/ADEChallengeData2016:/mdata/ade/ADEChallengeData2016/ \
    -v ~/itpsea4data/dataset/ADE20K_2021_17_01:/mdata/ade20k_full \
    -v ~/itpsea4data/dataset/coco_stuff164k:/mdata/coco_stuff164k \
    -v ~/itpsea4data/dataset/imagenet22k_azcopy:/mdata/imagenet21k \
    -v /mnt:/mnt \
    -u $(id -u):$(id -g) \
    "${IMAGE}" \
    bash -c "$CMD"

# run this file by: bash local.sh rel_path/to/config
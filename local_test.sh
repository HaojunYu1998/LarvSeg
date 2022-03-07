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
ln -s /mdata/ade /data/ade
ln -s /mdata/ade20k_full /data/ade20k_full
ln -s /mdata/coco /data/coco

cd /workspace
if $TASK_IS_EXPERIMENT; then
    bash tools/dist_test.sh $QUOTED_ARGS
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
    -v ~/itesea4data/datasets/ADEChallengeData2016:/mdata/ade/ADEChallengeData2016/ \
    -v ~/itesea4data/datasets/ADE20K_2021_17_01:/mdata/ade20k_full \
    -v ~/itesea4data/datasets/coco2017:/mdata/coco \
    -v /mnt:/mnt \
    -u $(id -u):$(id -g) \
    "${IMAGE}" \
    bash -c "$CMD"


# run this file by: bash local.sh rel_path/to/config
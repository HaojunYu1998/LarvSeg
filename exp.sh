#!/bin/bash

# NOTE: this script should be run from inside the container!
#
# In entrypoint of AML experiment, invoke this script like:
# bash exp.sh path/to/config.py --auto-resume

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo TEST
ls /haojun_storage
ls /haojun_storage/dataset
ls /haojun_storage/dataset/ADEChallengeData2016
ls /haojun_storage/dataset/imagenet22k_azcopy

# Change to the mounted path inside the container
if [[ "$@" =~ .*ade20kfull.* ]]; then
    ROOT=/haojun_storage/dataset/ADE20K_2021_17_01
elif [[ "$@" =~ .*ade20k.* ]]; then
    ROOT=/haojun_storage/dataset/ADEChallengeData2016
# elif [[ "$@" =~ .*cityscapes.* ]]; then
#     ROOT=/haojun_storage/dataset/original_cityscapes_minimal
elif [[ "$@" =~ .*pascal_context.* ]]; then
    ROOT=/haojun_storage/dataset/VOCdevkit/VOC2010
elif [[ "$@" =~ .*coco-stuff164k.* ]]; then
    ROOT=/haojun_storage/dataset/coco_stuff164k
elif [[ "$@" =~ .*imagenet21k.* ]]; then
    ROOT=/haojun_storage/dataset/imagenet22k_azcopy
else
    echo unsupported config $@
    exit 1
fi

echo $ROOT

echo bash tools/dist_train.sh "$@" --auto-resume\
    --options \
    data.train.data_root=$ROOT \
    data.test.data_root=$ROOT \
    data.val.data_root=$ROOT
bash tools/dist_train.sh "$@" --auto-resume\
    --options \
    data.train.data_root=$ROOT \
    data.test.data_root=$ROOT \
    data.val.data_root=$ROOT
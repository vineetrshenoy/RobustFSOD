#!/bin/bash
EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=/cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/MSRA/R-101.pkl                           
IMAGENET_PRETRAIN_TORCH=/cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  
TAGS=$2


python3 train_net.py --num-gpus 4 --config-file configs/coco/base.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
           OUTPUT_DIR ${SAVEDIR}/base
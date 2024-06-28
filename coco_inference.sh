#!/bin/bash

IMAGENET_PRETRAIN=/cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/MSRA//R-101.pkl                           
IMAGENET_PRETRAIN_TORCH=/cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  

CONFIG_PATH=/cis/home/vshenoy/FewShotObjectDetection/coco_train/checkpoints/coco/coco-8867b-05-10/mfdc_gfsod_novel/tfa-like/5shot_seed0/config.yaml
WEIGHTS=/cis/home/vshenoy/FewShotObjectDetection/coco_train/checkpoints/coco/coco-8867b-05-10/mfdc_gfsod_novel/tfa-like/5shot_seed0/model_final.pth
OUTPUT_DIR=/cis/net/r22a/data/vshenoy/mfdc-data/neurips2023paper/coco5shot_seed0_orig_images
#CONFIG_PATH=/cis/home/vshenoy/FewShotObjectDetection/coco_train/checkpoints/coco/coco-05-11-test/base/config.yaml
#WEIGHTS=/cis/home/vshenoy/FewShotObjectDetection/coco_train/checkpoints/coco/coco-05-11-test/base/model_final.pth
#OUTPUT_DIR=/cis/net/r22a/data/vshenoy/mfdc-data/neurips2023paper/base_proposals_fiddle


#CONFIG_PATH=/cis/home/vshenoy/FewShotObjectDetection/MFDC_1_baseline/checkpoints/coco/coco_base_test/base/config.yaml
#WEIGHTS=/cis/home/vshenoy/FewShotObjectDetection/MFDC_1_baseline/checkpoints/coco/coco_base_test/base/model_final.pth
#OUTPUT_DIR=/cis/net/r22a/data/vshenoy/mfdc-data/neurips2023paper/base_proposals_withrpnloss

#CONFIG_PATH=/cis/home/vshenoy/FewShotObjectDetection/MFDC_1_baseline/checkpoints/coco/withmodel/mfdc_gfsod_novel/tfa-like/5shot_seed0/config.yaml
#WEIGHTS=/cis/home/vshenoy/FewShotObjectDetection/MFDC_1_baseline/checkpoints/coco/withmodel/mfdc_gfsod_novel/tfa-like/5shot_seed0/model_final.pth
#OUTPUT_DIR=/cis/net/r22a/data/vshenoy/mfdc-data/neurips2023paper/coco5shot_seed0_baseline

python3 train_net.py --num-gpus 4 --config-file ${CONFIG_PATH} --eval-only --opts MODEL.WEIGHTS ${WEIGHTS} OUTPUT_DIR ${OUTPUT_DIR} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.0


#python3 train_net.py --num-gpus 4 --config-file ${CONFIG_PATH} --eval-only --opts MODEL.WEIGHTS ${WEIGHTS} OUTPUT_DIR ${OUTPUT_DIR} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.4


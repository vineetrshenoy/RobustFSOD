#!/usr/bin/env bash

EXPNAME=$1
SAVEDIR=checkpoints/voc/${EXPNAME}
IMAGENET_PRETRAIN=/cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/MSRA/R-101.pkl                           
IMAGENET_PRETRAIN_TORCH=/cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  
SPLIT_ID=$2
TAGS=$3

#CUDA_VISIBLE_DEVICES=5,6,7,8 python3 train_net.py --num-gpus 4 --config-file configs/voc/base${SPLIT_ID}.yaml     \
#    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
#           OUTPUT_DIR ${SAVEDIR}/base${SPLIT_ID}


#CUDA_VISIBLE_DEVICES=5,6,7,8 python3 tools/model_surgery.py --dataset voc --method randinit                                \
#    --src-path ${SAVEDIR}/base${SPLIT_ID}/model_final.pth                    \
#    --save-dir ${SAVEDIR}/base${SPLIT_ID} 

BASE_WEIGHT=${SAVEDIR}/base${SPLIT_ID}/model_reset_surgery.pth
#
#

#python3 tools/create_config.py --dataset voc --config_root configs/voc               \
#            --shot 1 --seed 0 --setting 'gfsod' --split 1
#
#
#CONFIG_PATH=configs/voc/mfdc_fsod_novel1_1shot_seed1.yaml
#OUTPUT_DIR=checkpoints/voc/vocR101-1/mfdc_gfsod_novel1/tfa-like/1shot_seed0
##
#CUDA_VISIBLE_DEVICES=1 python3 -m pdb train_net.py --num-gpus 1 --config-file configs/voc/mfdc_gfsod_novel1_1shot_seed0.yaml \
#            --opts MODEL.WEIGHTS checkpoints/voc/vocR101-1-test/base1/model_reset_surgery.pth OUTPUT_DIR checkpoints/voc/vocR101-1-test/mfdc_gfsod_novel1/tfa-like/1shot_seed0\
#                   TEST.PCB_MODELPATH /cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth SOLVER.WEIGHT_DECAY 0.00004

for seed in 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/mfdc_gfsod_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/mfdc_gfsod_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 train_net.py --dist-url tcp://0.0.0.0:25645 --num-gpus 4 --config-file ${CONFIG_PATH}                            \
            --tags ${TAGS} --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SOLVER.WEIGHT_DECAY 0.00004
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done

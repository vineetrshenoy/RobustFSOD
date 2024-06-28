EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=/cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/MSRA//R-101.pkl                           
IMAGENET_PRETRAIN_TORCH=/cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  
TAGS=$2

#python3 train_net.py --num-gpus 4 --config-file configs/coco/base50.yaml --opts OUTPUT_DIR checkpoints/coco/cocoR50TEMP/base


#python3 train_net.py --num-gpus 4 --config-file configs/coco/base.yaml --opts OUTPUT_DIR checkpoints/coco/cocoR101TEMP/base
#python3 train_net.py --num-gpus 4 --config-file configs/coco/base.yaml     \
#    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
#           OUTPUT_DIR ${SAVEDIR}/base
#
#
#python3 tools/model_surgery.py --dataset coco --method randinit                        \
#    --src-path ${SAVEDIR}/base/model_final.pth                         \
#    --save-dir ${SAVEDIR}/base
BASE_WEIGHT=${SAVEDIR}/base/model_reset_surgery.pth
#
#
for seed in 0 1 2 3 4
do
    for shot in 30
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
            --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/mfdc_gfsod_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/mfdc_gfsod_novel/tfa-like/${shot}shot_seed${seed}
        python3 train_net.py --num-gpus 4 --config-file ${CONFIG_PATH} --tags ${TAGS}                    \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
        #rm ${OUTPUT_DIR}/model_final.pth
    done
done
#
#
##python3 train_net.py --num-gpus 4 --config-file configs/coco/mfdc_gfsod_novel_3shot_seed0.yaml  --opts MODEL.WEIGHTS checkpoints/coco/coco_verify/base/model_reset_surgery.pth OUTPUT_DIR checkpoints/coco/coco_verify  TEST.PCB_MODELPATH /cis/net/io62a/data/vshenoy/tfa-data/datasets/ImageNetPretrained/MSRA/resnet101-5d3b4d8f.pth
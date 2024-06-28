#!/usr/bin/env bash
bash voc_train.sh voc-1-supcon-all125 1 supcon-all-test125
bash voc_train.sh voc-2-supcon-all 2 supcon-all-test
bash voc_train.sh voc-1-supcon-all 3 supcon-all-test
bash coco_train.sh coco-03-16-allgeom-wnoise allgeom_wnoise5

bash coco_train.sh coco_test coco_test

CUDA_VISIBLE_DEVICES=0,1,2,3 bash voc_train_first_half.sh voc-2-supcon-all125 2 voc-2-supcon-all125
CUDA_VISIBLE_DEVICES=1,2,3,5 bash voc_train_second_half.sh voc-3-supconall125 3 voc-3-supconall125


bash coco_train_first_half.sh coco-05-04 coco-05-04
bash coco_train_second_half.sh coco-04-26 coco-04-26

bash voc_train_first_half.sh voc-2-05-02a 2 voc-2-05-02a
bash voc_train_second_half.sh voc-3-05-02a 3 voc-3-05-02a



bash voc_train.sh voc-1-supconall-median 1 voc-1-supconall-median
bash voc_train.sh vocR101-1-test 1 temp


bash voc_train.sh voc-1-05-04 1 05-04
CUDA_VISIBLE_DEVICES=6,7,8,9 bash voc_train.sh voc-2-05-04 2 05-04
bash voc_train.sh voc-3-05-04 3 05-04
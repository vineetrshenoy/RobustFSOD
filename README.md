# Robust Feature Space Organization with Distillation for Few-Shot Object Detection, ICPR 2024

(To be updated)

This codebase was used to generate the results in 
```
Shenoy, Vineet R., and Rama Chellappa. "Robust Feature Space Organization with Distillation for Few-Shot Object Detection." International Conference on Pattern Recognition. Cham: Springer Nature Switzerland, 2024.
```


This repo is built upon [DeFRCN](https://github.com/er-muyue/DeFRCN), where you can download the datasets and the pre-trained weights.


## Requirements
Python == 3.7.10

Pytorch == 1.6.0

Torchvision == 0.7.0

CUDA == 10.1


## File Structure
```
    ├── weight/                   
    |   ├── R-101.pkl              
    |   └── resnet101-5d3b4d8f.pth   
    └── datasets/
        ├── coco/           
        │   ├── annotations/
        │   ├── train2014/
        │   └── val2014/
        ├── cocosplit/
        ├── VOC2007/            
        │   ├── Annotations/
        │   ├── ImageSets/
        │   └── JPEGImages/
        ├── VOC2012/            
        │   ├── Annotations/
        │   ├── ImageSets/
        │   └── JPEGImages/
        └── vocsplit/
```


## Training and Evaluation
* For COCO
```
bash coco_train.sh experiment_name
```


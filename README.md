# :zap:FastestDet:zap:[![DOI](https://zenodo.org/badge/508635170.svg)](https://zenodo.org/badge/latestdoi/508635170)
![image](https://github.com/dog-qiuqiu/FastestDet/blob/main/data/data.png)
* ***Faster! Stronger! Simpler!***
* ***It has better single core reasoning performance and simpler feature map post-processing than Yolo-fastest***
* ***In the ARM CPU of RK3568, the single core reasoning performance is 50% higher than Yolo-fastest***
* ***The coco evaluation index increased by 3.8% compared with the map0.5 of Yolo-fastest***
* ***算法介绍：https://zhuanlan.zhihu.com/p/536500269 交流qq群:1062122604***
# Evaluating indicator/Benchmark
Network|COCO mAP(0.5)|Resolution|Run Time(4xCore)|Run Time(1xCore)|FLOPs(G)|Params(M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[Yolo-FastestV1.1](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|24.40 %|320X320|26.60 ms|75.74 ms|0.252|0.35M
[Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2/tree/main/modelzoo)|24.10 %|352X352|23.8 ms|68.9 ms|0.212|0.25M
FastestDet|27.8%|512X512|21.51ms|34.62ms|*|0.25M

* ***Test platform RK3568 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
# Improvement
* Anchor-Free
* Single scale detector head
* Cross grid multiple candidate targets
* Dynamic positive and negative sample allocation
# How to use
## Dependent installation
* PiP(Note pytorch CUDA version selection)
  ```
  pip install -r requirements.txt
  ```
## Test
* Picture test
  ```
  python3 test.py --yaml configs/config.yaml --weight weights/weight_AP05\:0.278_280-epoch.pth --img data/3.jpg
  ```
<div align=center>
<img src="https://github.com/dog-qiuqiu/FastestDet/blob/main/result.png"> />
</div>

## How to train
### Building data sets(The dataset is constructed in the same way as darknet yolo)
* The format of the data set is the same as that of Darknet Yolo, Each image corresponds to a .txt label file. The label format is also based on Darknet Yolo's data set label format: "category cx cy wh", where category is the category subscript, cx, cy are the coordinates of the center point of the normalized label box, and w, h are the normalized label box The width and height, .txt label file content example as follows:
  ```
  11 0.344192634561 0.611 0.416430594901 0.262
  14 0.509915014164 0.51 0.974504249292 0.972
  ```
* The image and its corresponding label file have the same name and are stored in the same directory. The data file structure is as follows:
  ```
  .
  ├── train
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  └── val
      ├── 000043.jpg
      ├── 000043.txt
      ├── 000057.jpg
      ├── 000057.txt
      ├── 000070.jpg
      └── 000070.txt
  ```
* Generate a dataset path .txt file, the example content is as follows：
  
  train.txt
  ```
  /home/qiuqiu/Desktop/dataset/train/000001.jpg
  /home/qiuqiu/Desktop/dataset/train/000002.jpg
  /home/qiuqiu/Desktop/dataset/train/000003.jpg
  ```
  val.txt
  ```
  /home/qiuqiu/Desktop/dataset/val/000070.jpg
  /home/qiuqiu/Desktop/dataset/val/000043.jpg
  /home/qiuqiu/Desktop/dataset/val/000057.jpg
  ```
* Generate the .names category label file, the sample content is as follows:
 
  category.names
  ```
  person
  bicycle
  car
  motorbike
  ...
  
  ```
* The directory structure of the finally constructed training data set is as follows:
  ```
  .
  ├── category.names        # .names category label file
  ├── train                 # train dataset
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  ├── train.txt              # train dataset path .txt file
  ├── val                    # val dataset
  │   ├── 000043.jpg
  │   ├── 000043.txt
  │   ├── 000057.jpg
  │   ├── 000057.txt
  │   ├── 000070.jpg
  │   └── 000070.txt
  └── val.txt                # val dataset path .txt file

  ```
### Build the training .yaml configuration file
* Reference./configs/config.yaml
```
DATASET:
  TRAIN: "/home/qiuqiu/Desktop/coco2017/train2017.txt"  # Train dataset path .txt file
  VAL: "/home/qiuqiu/Desktop/coco2017/val2017.txt"      # Val dataset path .txt file 
  NAMES: "dataset/coco128/coco.names"                   # .names category label file
MODEL:
  NC: 80                                                # Number of detection categories
  INPUT_WIDTH: 512                                      # The width of the model input image
  INPUT_HEIGHT: 512                                     # The height of the model input image
TRAIN:
  LR: 0.001                                             # Train learn rate
  THRESH: 0.25                                          # ？？？？
  WARMUP: true                                          # Trun on warm up
  BATCH_SIZE: 64                                        # Batch size
  END_EPOCH: 350                                        # Train epichs
  MILESTIONES:                                          # Declining learning rate steps
    - 150
    - 250
    - 300
```
### Train
* Perform training tasks
  ```
  python3 train.py --yaml configs/config.yaml
  ```
### Evaluation
* Calculate map evaluation
  ```
  python3 eval.py --yaml configs/config.yaml --weight weights/weight_AP05\:0.278_280-epoch.pth
  ```
* COCO2017 evaluation
  ```
  creating index...
  index created!
  creating index...
  index created!
  Running per image evaluation...
  Evaluate annotation type *bbox*
  DONE (t=30.85s).
  Accumulating evaluation results...
  DONE (t=4.97s).
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.140
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.278
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.128
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.103
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.232
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.157
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.225
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.032
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.201
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.359

  ```
# Deploy
## NCNN
* Waiting for update
# Citation
* If you find this project useful in your research, please consider cite:
```
@misc{=FastestDet,
    title={FastestDet: Ultra lightweight anchor-free real-time object detection algorithm.},
    author={xuehao.ma},
    howpublished = {\url{https://github.com/dog-qiuqiu/FastestDet}},
    year={2022}
}
```
# Reference
* https://github.com/Tencent/ncnn

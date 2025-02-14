APE Learning for Person Re-identification
=========

Data
---------
We use APE learning on four Re-ID datasets:    
- Market1501
- DukeMTMC
- CUHK03(np-detected)
- MSMT17

your dataset tree should be like this:

```
dataset/
|-- DukeMTMC-reID
|   |-- CITATION.txt
|   |-- LICENSE_DukeMTMC-reID.txt
|   |-- LICENSE_DukeMTMC.txt
|   |-- README.md
|   |-- bounding_box_test
|   |-- bounding_box_train
|   `-- query
|-- Market-1501-v15.09.15
|   |-- bounding_box_test
|   |-- bounding_box_train
|   |-- gt_bbox
|   |-- gt_query
|   |-- query
|   `-- readme.txt
|-- cuhk03
|   |-- cuhk03_new_protocol_config_detected.mat
|   |-- cuhk03_new_protocol_config_labeled.mat
|   |-- cuhk03_release
|   |-- images_detected
|   |-- images_labeled
|   |-- splits_classic_detected.json
|   |-- splits_classic_labeled.json
|   |-- splits_new_detected.json
|   `-- splits_new_labeled.json
`-- msmt17
    |-- bounding_box_test
    |-- bounding_box_train
    `-- query
```
The evaluation criteria Market1501 and DukeMTMC are provided by the baseline.     
We use the new training/test protocal for CUHK03 ([CUHK03-NP](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)), which can calculate mAP and CMC similar to the above two datasets.

Train
---------
You can run the `./*.sh` files according to the different datasets directly or run these codes in the `.sh` files as follows:
1. Market1501   

```cpp
python3 tools/train.py --config_file='configs/APE_Learning.yml'  OUTPUT_DIR "/data/Checkpoints/ReID_Strong_BL/Market1501" LOG_NAME "log_test.txt" OURS.ALPHA "20.0" OURS.BETA "0.5" MODEL.DEVICE_ID "'0'" MODEL.ADJUST_LR "off"   MODEL.METRIC_LOSS_TYPE "ours" DATALOADER.SAMPLER "ours" MODEL.NECK "APE" DATASETS.NAMES "'market1501'"  INPUT.RE_PROB "0.7"  MODEL.LAST_STRIDE "1" DATALOADER.NUM_INSTANCE "8" SOLVER.BASE_LR "3.5e-4" SOLVER.WARMUP_ITERS "0" MODEL.IF_TRIPLET "no"
```

2. DukeMTMC   

```cpp
python3 tools/train.py --config_file='configs/APE_Learning.yml' OUTPUT_DIR "/data/Checkpoints/ReID_Strong_BL/Duke" LOG_NAME "log_test.txt" OURS.ALPHA "18.0" OURS.BETA "0.5" MODEL.DEVICE_ID "'1'" MODEL.ADJUST_LR "off" MODEL.METRIC_LOSS_TYPE "ours" DATALOADER.SAMPLER "ours" MODEL.NECK "APE" DATASETS.NAMES "'dukemtmc'"  INPUT.RE_PROB "0.5"  MODEL.LAST_STRIDE "1" DATALOADER.NUM_INSTANCE "8" SOLVER.BASE_LR "3.5e-4" SOLVER.WARMUP_ITERS "0" MODEL.IF_TRIPLET "no"
```

3. CUHK03-Detected    

```cpp
python3 tools/train.py --config_file='configs/APE_Learning.yml' OUTPUT_DIR "/data/Checkpoints/ReID_Strong_BL/cuhk03" LOG_NAME "log_test.txt" OURS.ALPHA "8.0" OURS.BETA "0.5" MODEL.DEVICE_ID "'1'" MODEL.ADJUST_LR "off" MODEL.METRIC_LOSS_TYPE "ours" DATALOADER.SAMPLER "ours" MODEL.NECK "APE" DATASETS.NAMES "'cuhk03'"  INPUT.RE_PROB "0.5"  MODEL.LAST_STRIDE "1" DATALOADER.NUM_INSTANCE "8" SOLVER.BASE_LR "3.5e-4" SOLVER.WARMUP_ITERS "0" MODEL.IF_TRIPLET "no"
```

4. Main Parameters in training
```cpp
'--config_file': the base configuration file
OUTPUT_DIR: the output path to save model and log
LOG_NAME: the name of the output log
OURS.ALPHA: the hyper-parameter tn in Equation (21)
OURS.BETA: the hyper-parameter \alpha in Equation (21)
MODEL.DEVICE_ID: the gpu number to train the network
MODEL.ADJUST_LR: to determine how to adjust the learning rate
MODEL.METRIC_LOSS_TYPE: the type of loss
DATALOADER.SAMPLER: the sampling stategy
MODEL.NECK: the nect architecture of network ("APE", "bnneck", "no")
DATASETS.NAMES: the dataset
INPUT.RE_PROB: the probability of random erasing
MODEL.LAST_STRIDE: the stride of the last convolutional layer
DATALOADER.NUM_INSTANCE: the number of instance in a batch
SOLVER.BASE_LR: the base/initial learning rate
SOLVER.WARMUP_ITERS: controlling warmup of the learning rate ("0" means no warmup)
MODEL.IF_TRIPLET: "1" means training with the combination of APE loss and triplet loss; "0" means training with the individual APE loss
```

Notes
---------
The codes are expanded on a [Re-ID Strong Baseline](https://github.com/michuanhaohao/reid-strong-baseline). Thus, the baseline has some authors information, which are not belong to any author in our paper.

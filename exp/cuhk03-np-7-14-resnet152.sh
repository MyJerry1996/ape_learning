# APE Loss for CUHK03-Detected
python3 tools/train.py --config_file='configs/APE_Learning.yml' \
    OUTPUT_DIR "./outputs/cuhk03/7-14-resnet152" \
    LOG_NAME "log_test.txt" \
    OURS.ALPHA "8.0" \
    OURS.BETA "0.5" \
    MODEL.DEVICE_ID "'1'" \
    MODEL.ADJUST_LR "off" \
    MODEL.METRIC_LOSS_TYPE "ours" \
    DATALOADER.SAMPLER "ours" \
    MODEL.NECK "APE" \
    DATASETS.NAMES "'cuhk03'"  \
    INPUT.RE_PROB "0.5"  \
    MODEL.LAST_STRIDE "1" \
    DATALOADER.NUM_INSTANCE "8" \
    SOLVER.BASE_LR "3.5e-4" \
    SOLVER.WARMUP_ITERS "0" \
    MODEL.IF_TRIPLET "no" \
    MODEL.PRETRAIN_PATH "./weights/resnet152-b121ed2d.pth"  \
    MODEL.SAVE_NAME "resnet152_ours" \
    MODEL.NAME "resnet152"
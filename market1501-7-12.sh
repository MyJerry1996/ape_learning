python3 tools/train.py --config_file='configs/APE_Learning.yml'  \
    OUTPUT_DIR "./outputs/market1501/7-12" \
    LOG_NAME "log_test.txt" \
    OURS.ALPHA "20.0" \
    OURS.BETA "0.5" \
    MODEL.DEVICE_ID "'0'" \
    MODEL.ADJUST_LR "off"  \
    MODEL.METRIC_LOSS_TYPE "ours" \
    DATALOADER.SAMPLER "ours" \
    MODEL.NECK "APE" \
    DATASETS.NAMES "'market1501'" \
    INPUT.RE_PROB "0.7" \
    MODEL.LAST_STRIDE "1" \
    DATALOADER.NUM_INSTANCE "8" \
    SOLVER.BASE_LR "3.5e-4" \
    SOLVER.WARMUP_ITERS "0" \
    MODEL.IF_TRIPLET "no"
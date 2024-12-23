# Hyperparameter Settings
METHOD="derpp"     # "er" or "derpp"
SPARSE=0.75
GPU_ID=0
DEVICE="Odroid"
PATH_TO_SPARCL=/home/odroid/hs/SparCL-NCM # change to your own path

DATASET="seq-cifar100"
GLOBAL_BATCH_SIZE="32"

# magnitude-based 1 shot retraining
ARCH="resnet" # 
DEPTH="18"
PRUNE_ARGS="--sp-retrain --sp-prune-before-retrain"
LOAD_CKPT="XXXXX.pth.tar"     # automatically train from scratch if the given checkpoint model is not found
INIT_LR="0.03"
EPOCHS="150"
WARMUP="8"

SPARSITY_TYPE="irregular"
MASK_UPDATE_DECAY_EPOCH="5-45"
SP_MASK_UPDATE_FREQ="5"

REMOVE_N=3000
RM_EPOCH=20
GRADIENT=0.8
ITER=10
# REMOVE_N=0
# RM_EPOCH=-1
# GRADIENT=1
# ITER=1

SAVE_FOLDER="checkpoints/resnet18/paper/gradient_effi/mutate_irr/${DATASET}/buffer_${BUFFER_SIZE}/"
cd $PATH_TO_SPARCL
mkdir -p ${SAVE_FOLDER}

# ------- for overall sparsity ----------
# ------- check retrain.py for more information ----------
LOWER_BOUND="${SPARSE}-$(awk "BEGIN {printf \"%.2f\", ${SPARSE}+0.01}")-${SPARSE}"
UPPER_BOUND="$(awk "BEGIN {printf \"%.2f\", ${SPARSE}-0.01}")-${SPARSE}-${SPARSE}"

CONFIG_FILE="./profiles/resnet18_cifar/irr/resnet18_${SPARSE}.yaml"
REMARK="irr_${SPARSE}_mut"
LOG_NAME="${SPARSE}_${METHOD}_${GRADIENT}"
PKL_NAME="irr_${SPARSE}_mut_RM_${REMOVE_N}_${RM_EPOCH}"

SEED=42
for BUFFER_SIZE in 100 200 300 400 500
do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u main_sparse_train_w_data_gradient_efficient.py \
        --arch ${ARCH} --depth ${DEPTH} --optmzr sgd --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler cosine --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --dataset ${DATASET} --seed ${SEED} --upper-bound ${UPPER_BOUND} --lower-bound ${LOWER_BOUND} --mask-update-decay-epoch ${MASK_UPDATE_DECAY_EPOCH} --sp-mask-update-freq ${SP_MASK_UPDATE_FREQ} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} \
        --log-filename=${SAVE_FOLDER}/seed_${SEED}_${LOG_NAME}.txt --buffer-size=$BUFFER_SIZE --replay_method $METHOD --buffer_weight 0.1 --buffer_weight_beta 0.5 \
        --use_cl_mask --gradient_sparse=$GRADIENT --remove-n=$REMOVE_N --keep-lowest-n 0 --remove-data-epoch=$RM_EPOCH --output-dir ${SAVE_FOLDER} --output-name=${PKL_NAME} --iter $ITER --ncm --gradient_efficient_mix \
        --device $DEVICE --sparse_ratio $SPARSE --rand-seed
done
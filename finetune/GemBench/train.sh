cd /home/wzh/BridgeVLA/finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0

cd GemBench


port=15559
GPUS_PER_NODE=2
NNODES=1
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$port \
    train.py \
    $@ 


# bash train.sh --exp_cfg_path  configs/colosseum_config.yaml \
#               --exp_note debug \
#               --freeze_vision_tower \
#               --log_dir PATH_TO_LOG_DIR \
#               --load_pretrain \
#               --pretrain_path  PATH_TO_PRETRAINED_MODEL
# 设置 Hugging Face 缓存路径
export HF_HOME="/home/BridgeVLA/huggingface"



cd /home/wzh/BridgeVLA/finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0


cd /home/wzh/BridgeVLA/finetune/Real
echo "所有传入的参数：$@"

# torchrun    --nnodes  1 \
#             --node_rank 0 \
#             --nproc_per_node 1 \
#             --master_addr 127.0.0.1  \
#             --master_port 12345 \
#             train_torchrun_real.py \
#             $@ 
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --nnodes=1 --node_rank=0 --master_port 15555 --nproc_per_node=4 train.py $@ 



# train with test data
# nohup bash  /home/wzh/BridgeVLA/finetune/Real/train.sh    --exp_cfg_path configs/real.yaml --exp_note a100_dobot_open_door --layer_index -1 --freeze_vision_tower --load_pretrain --pretrain_path  /home/BridgeVLA/pretrained_ckpt/pretrain/one_image_layer1_pretrain_3824  --data_folder /home/BridgeVLA/data/0616_open_the_door --test_data_folder /home/BridgeVLA/data/0616_open_the_door_test  --ep_per_task  25 &

# train with arm flag
# nohup bash  /home/wzh/BridgeVLA/finetune/Real/train.sh    --exp_cfg_path configs/real.yaml --exp_note a100_dobot_open_door_arm_flag --layer_index -1 --freeze_vision_tower --load_pretrain --pretrain_path  /home/BridgeVLA/pretrained_ckpt/pretrain/one_image_layer1_pretrain_3824  --data_folder /home/BridgeVLA/data/20250627 --test_data_folder /home/BridgeVLA/data/20250627  --ep_per_task  25 --output_arm_flag &

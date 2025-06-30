import os
import subprocess
import time
import tqdm
import random
import yaml
import argparse
import time
from collections import defaultdict
from contextlib import redirect_stdout
# from accelerate import PartialState
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
# Set NCCL environment variables
# os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 或者使用你的实际网络接口名称
# os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用 InfiniBand
# os.environ['NCCL_P2P_DISABLE'] = '1'  # 禁用 P2P 通信
import sys
# 增加地址到sys.path
sys.path.append('/home/wzh/BridgeVLA/finetune/')
print('sys.path:',sys.path)
import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod

from bridgevla.mvt.mvt import MVT
from bridgevla.models.bridgevla_agent import print_eval_log, print_loss_log
from bridgevla.utils.rvt_utils import (
    get_num_feat,
    load_agent,
)
from utils.peract_utils import (
    CAMERAS_REAL,
    SCENE_BOUNDS_real,
    IMAGE_SIZE,
)
import socket
from huggingface_hub import login
# login(token='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')
import warnings
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from real_dataset import Real_Dataset
import datetime

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))  # 0 will make the OS choose an available port
        return s.getsockname()[1]




def create_dataloader(dataset, rank, world_size, batch_size, num_workers, use_distributed=True):
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True  # 是否打乱数据
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True, # 丢弃最后一个不完整批次
            pin_memory=True  # 加速数据加载
        )
        return dataloader, sampler
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        return dataloader, None

def reduce_value(value, average=True):
    # 跨进程汇总标量值
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / dist.get_world_size() if average else tensor.item()


def train(agent, data_loader, cameras=["3rd","wrist"], rank=0, update_dpo=False, dpo_beta=0.1):
    agent.train()
    if rank == 0:
        print(f"You are using {cameras} for training")
        if update_dpo:
            print(f"Using DPO training mode with beta={dpo_beta}")
        else:
            print("Using standard training mode")
    def move_tensors_to_device(d, device):
        if isinstance(d, dict):
            return {k: move_tensors_to_device(v, device) if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in d.items()}
        return d
    iteration=0
    epoch_losses={}
    
    # 确保所有进程同步
    dist.barrier()
    
    for raw_batch in  tqdm.tqdm(data_loader, disable=(rank != 0), position=0, leave=True):
        iteration+=1
        batch = move_tensors_to_device(raw_batch, agent._device)
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"]=[[[item]] for item in raw_batch["lang_goal"]]
        
        if update_dpo:
            # DPO训练模式：需要正例和负例数据
            # 假设batch中已经包含了正例和负例数据
            update_args = {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": False,
                "cameras": cameras,
                "beta": dpo_beta,
            }
            out = agent.dpo_update_real(**update_args)
        else:
            # 标准训练模式
            update_args = {
                "cameras": cameras,
            }
            update_args.update(
                {
                    "replay_sample": batch,
                    "backprop": True,
                    "reset_log": (iteration == 0),
                    "eval_log": False,
                }
            )
            out = agent.update_real(**update_args)
            
        if epoch_losses=={}:
            epoch_losses = {key: [] for key in out.keys()}
        for key in epoch_losses:
            loss_value = out[key]
            # 汇总所有进程的 loss
            reduced_loss = reduce_value(loss_value, average=True)
            epoch_losses[key].append(reduced_loss)
    
    # 确保所有进程同步
    dist.barrier()
    
    if rank == 0:
        time.sleep(10)
        log = print_loss_log(agent)
    
    # 确保所有进程同步
    dist.barrier()
    
    avg_losses = {key: sum(values)/len(values) for key, values in epoch_losses.items()}
    return avg_losses

def evaluate(agent, data_loader, cameras=["3rd","wrist"], rank=0, update_dpo=False, dpo_beta=0.1):
    agent.eval()
    if rank == 0:
        print(f"You are using {cameras} for evaluation")
        if update_dpo:
            print(f"Using DPO evaluation mode with beta={dpo_beta}")
        else:
            print("Using standard evaluation mode")
    
    def move_tensors_to_device(d, device):
        if isinstance(d, dict):
            return {k: move_tensors_to_device(v, device) if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in d.items()}
        return d
    
    epoch_losses = {}
    
    # 确保所有进程同步
    dist.barrier()
    
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(tqdm.tqdm(data_loader, disable=(rank != 0), position=0, leave=True)):
            batch = move_tensors_to_device(raw_batch, agent._device)
            batch["tasks"] = raw_batch["tasks"]
            batch["lang_goal"]=[[[item]] for item in raw_batch["lang_goal"]]
            
            if update_dpo:
                # DPO评估模式
                update_args = {
                    "replay_sample": batch,
                    "backprop": False,
                    "reset_log": (batch_idx == 0),
                    "eval_log": True,
                    "cameras": cameras,
                    "beta": dpo_beta,
                }
                out = agent.dpo_update_real(**update_args)
            else:
                # 标准评估模式
                update_args = {
                    "cameras": cameras,
                }
                update_args.update(
                    {
                        "replay_sample": batch,
                        "backprop": False,
                        "reset_log": (batch_idx == 0),  # 只在第一个批次时重置日志
                        "eval_log": True,
                    }
                )
                out = agent.update_real(**update_args)
                
            if epoch_losses == {}:
                epoch_losses = {key: [] for key in out.keys()}
            for key in epoch_losses:
                loss_value = out[key]
                # 汇总所有进程的 loss
                reduced_loss = reduce_value(loss_value, average=True)
                epoch_losses[key].append(reduced_loss)
    
    # 确保所有进程同步
    dist.barrier()

    avg_losses = {key: sum(values)/len(values) for key, values in epoch_losses.items()}
    return avg_losses



def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def get_time():
    import datetime
    # 获取当前时间
    now = datetime.datetime.now()
    # 获取月份、日期、小时、分钟
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    # 将它们拼接成一个字符串，格式为 'MM-DD-HH-MM'
    folder_name = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return folder_name


def get_logdir(cmd_args, exp_cfg,dist):
    log_dir = os.path.join(cmd_args.log_dir,"train" ,exp_cfg.exp_id,cmd_args.exp_note)
    if cmd_args.debug==True:
        log_dir = os.path.join(log_dir,"real_debug")

    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    trial_time=get_time()
    log_dir = os.path.join(log_dir,f"trial_real_{trial_time}")
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)



def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            print("Can not find RANK and WORLD_SIZE, Debug Mode")
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "9001"
            os.environ["LOCAL_RANK"] = "0"
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


def experiment(cmd_args):
    print("You are using:",cmd_args.cameras)
    setup_distributed()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print("local_rank:",local_rank)
    device_id = f"cuda:{local_rank}"

    torch.cuda.set_device(device_id)

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))
    
    ddp = int(os.environ['WORLD_SIZE']) > 1
    print(f"Total devices: {dist.get_world_size()}")
    if ddp:
        print(f"Running DDP on rank {dist.get_rank()}.")
    if cmd_args.lr != exp_cfg.peract.lr:
        warnings.warn(f"Warning: cmd_args.lr={cmd_args.lr} != exp_cfg.peract.lr={exp_cfg.peract.lr}")
        old_exp_cfg_peract_lr=cmd_args.lr
    else:
        old_exp_cfg_peract_lr = exp_cfg.peract.lr
    print("Your learning rate is:",old_exp_cfg_peract_lr)
    old_exp_cfg_exp_id = exp_cfg.exp_id
    
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"
    

    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.peract.transform_augmentation=True 
    exp_cfg.freeze()

    BATCH_SIZE_TRAIN = exp_cfg.bs
    print(f"BATCH_SIZE_TRAIN={BATCH_SIZE_TRAIN}")

    EPOCHS = exp_cfg.epochs
    data_folder = cmd_args.data_folder
    test_data_folder = cmd_args.test_data_folder if hasattr(cmd_args, 'test_data_folder') else None

    log_dir = get_logdir(cmd_args, exp_cfg, dist)
    t_start = time.time()
    
    # 创建训练数据集
    train_dataset = Real_Dataset(data_folder, device=device_id, cameras=cmd_args.cameras, ep_per_task=cmd_args.ep_per_task, output_arm_flag=cmd_args.output_arm_flag)
    print("Total tasks: ", train_dataset.num_tasks)
    print("Total trajectories: ", train_dataset.num_task_paths)
    print("Dataset Length: ", len(train_dataset))
    
    # 创建测试数据集（如果提供了测试数据文件夹）
    test_dataset = None
    if test_data_folder:
        test_dataset = Real_Dataset(test_data_folder, device=device_id, cameras=cmd_args.cameras, ep_per_task=cmd_args.ep_per_task, output_arm_flag=cmd_args.output_arm_flag)
        print("Test Dataset Length: ", len(test_dataset))

    train_dataloader, train_sampler = create_dataloader(train_dataset, rank, world_size, BATCH_SIZE_TRAIN, exp_cfg.num_workers, use_distributed=True)
    test_dataloader, test_sampler = create_dataloader(test_dataset, rank, world_size, BATCH_SIZE_TRAIN, exp_cfg.num_workers, use_distributed=True) if test_dataset else None
    
    t_end = time.time()
    if local_rank == 0:
        print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    # 创建保存最佳模型的目录
    best_models_dir = os.path.join(log_dir, "best_models")
    if dist.get_rank() == 0:
        os.makedirs(best_models_dir, exist_ok=True)
    
    # 用于记录最佳模型信息的文件
    best_models_info_path = os.path.join(best_models_dir, "best_models_info.txt")
    
    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)

    mvt_cfg.freeze()

    # for maintaining backward compatibility
    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
    )

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        output_arm_flag=cmd_args.output_arm_flag,
        **mvt_cfg,
    )

    backbone = backbone.to(device_id)
    backbone = DDP(
        backbone,
        device_ids=[local_rank],
        find_unused_parameters=True,
    )
    
    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        # add_lang=mvt_cfg.add_lang,
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS_real,
        cameras=CAMERAS_REAL,
        log_dir=f"{log_dir}/test_run/",
        # cos_dec_max_step=EPOCHS * len(train_dataloader),
        # use_scheduler=exp_cfg.use_scheduler,
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    freeze_names=["lm_head","embed_tokens"]
    if cmd_args.freeze_vision_tower:
        freeze_names.append("vision_tower")
        print("Freeze vision tower")
    if cmd_args.freeze_language_model:
        freeze_names.append("language_model")
        print("Freeze language model")
    for name, module in agent._network.named_modules():
        for freeze_name in freeze_names:
            if freeze_name in name:
                for param in module.parameters():
                    param.requires_grad = False
                break

    # 计算可训练参数的总量
    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    # 转换为 billion 并打印
    total_params_billion = total_params / 1e9  # 1e9 即 10^9

    print(f'Total trainable parameters: {total_params_billion:.2f} billion')
    
    agent.build(training=True, device=device_id)

    start_epoch = 0
    end_epoch = EPOCHS
    # if exp_cfg.resume != "":
    #     agent_path = exp_cfg.resume
    #     print(f"Recovering model and checkpoint from {exp_cfg.resume}")
    #     epoch = load_agent(agent_path, agent, only_epoch=False)
    #     start_epoch = epoch + 1
    dist.barrier()

    if dist.get_rank() == 0:
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
    # Initialize Logging =>> W&B
    if dist.get_rank() == 0:
        wandb.login(key="64ae865da9e03328b96e0202de725fe607d6dd12")
        if os.getenv('DEBUG', 'false').lower() == 'true' or cmd_args.debug:
            wandb.init(entity="940257123", project="bridgevla", name=os.path.dirname(log_dir),mode="disabled")
        else:
            wandb.init(entity="940257123", project="bridgevla", name=os.path.dirname(log_dir))

    print("Start training ...", flush=True)
    i = start_epoch
    
    while True:
        if i == end_epoch:
            break

        # 设置sampler的epoch以确保每个epoch的数据顺序不同
        if train_sampler is not None:
            train_sampler.set_epoch(i)
        if test_sampler is not None:
            test_sampler.set_epoch(i)

        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training on train dataset")
        dist.barrier()
        train_losses = train(agent, train_dataloader, rank=dist.get_rank(), cameras=cmd_args.cameras, update_dpo=cmd_args.update_dpo, dpo_beta=cmd_args.dpo_beta)
        
        # 确保所有进程同步
        dist.barrier()
        
        # 所有进程都参与评估，并且每隔eval_interval个epoch评估一次
        if test_dataloader and (i % cmd_args.eval_interval == 0 or i == end_epoch - 1):
            if dist.get_rank() == 0:
                print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Evaluating on test dataset")
            
            # 所有进程都参与评估
            eval_losses = evaluate(agent, test_dataloader, rank=dist.get_rank(), cameras=cmd_args.cameras, update_dpo=cmd_args.update_dpo, dpo_beta=cmd_args.dpo_beta)
            
            # 确保所有进程都完成评估
            dist.barrier()
            
            # 只在rank 0上记录结果和保存模型
            if dist.get_rank() == 0:
                # 记录训练和评估损失
                wandb.log(eval_losses, step=i)
                
                if eval_losses:
                    eval_loss = sum(eval_losses.values()) / len(eval_losses)  # 使用所有损失的平均值
                    
                    # 保存当前模型
                    current_model_path = os.path.join(best_models_dir, f"model_{i}.pth")
                    save_agent(agent, current_model_path, i)
                    
                    # 读取现有的最佳模型信息
                    best_models_info = []
                    if os.path.exists(best_models_info_path):
                        with open(best_models_info_path, 'r') as f:
                            for line in f:
                                epoch, loss, path = line.strip().split(',')
                                best_models_info.append((int(epoch), float(loss), path))
                    
                    # 添加当前模型信息
                    best_models_info.append((i, eval_loss, current_model_path))
                    
                    # 按损失排序并只保留最好的3个
                    best_models_info.sort(key=lambda x: x[1])
                    best_models_info = best_models_info[:3]
                    
                    # 更新最佳模型信息文件
                    with open(best_models_info_path, 'w') as f:
                        for epoch, loss, path in best_models_info:
                            f.write(f"{epoch},{loss},{path}\n")
                    
                    # 删除不在最佳列表中的模型文件
                    existing_models = set(os.path.join(best_models_dir, f"model_{epoch}.pth") for epoch, _, _ in best_models_info)
                    for file in os.listdir(best_models_dir):
                        if file.endswith('.pth') and os.path.join(best_models_dir, file) not in existing_models:
                            os.remove(os.path.join(best_models_dir, file))
                    
                    # 保存最新的模型
                    save_agent(agent, f"{log_dir}/model_last.pth", i)
                    
                    # 打印当前最佳模型信息
                    print("\nCurrent best models:")
                    for epoch, loss, path in best_models_info:
                        print(f"Epoch {epoch}: Loss = {loss:.4f}, Path = {path}")
        else:
            if dist.get_rank() == 0:
                wandb.log(train_losses, step=i)
                if i % 50 == 0 or i == end_epoch-1:
                    save_agent(agent, f"{log_dir}/model_{i}.pth", i)
                    save_agent(agent, f"{log_dir}/model_last.pth", i)
        
        # 确保所有进程同步
        dist.barrier()
        
        i += 1

    dist.barrier()
    if dist.get_rank() == 0:
        print("[Finish]")
        # 打印最终的最佳模型信息
        if os.path.exists(best_models_info_path):
            print("\nFinal best models:")
            with open(best_models_info_path, 'r') as f:
                for line in f:
                    epoch, loss, path = line.strip().split(',')
                    print(f"Epoch {epoch}: Loss = {loss:.4f}, Path = {path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--mvt_cfg_path", type=str, default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/real.yaml")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--exp_note", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="/home/wzh/BridgeVLA/finetune/Real/logs")
    parser.add_argument("--data_folder", type=str, default="/home/wzh/BridgeVLA/finetune/Real/data/0616_open_the_door")
    parser.add_argument("--test_data_folder", type=str, default=None, help="Path to test dataset folder")
    parser.add_argument("--eval_interval", type=int, default=5, help="Interval between evaluations in epochs")
    
    parser.add_argument("--with-eval", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--palligemma_type", type=int, default=1)
    parser.add_argument("--layer_index", type=int, default=-1)
    parser.add_argument("--ep_per_task", type=int, default=5)
    parser.add_argument("--layer_concat", action="store_true",default=False)
    parser.add_argument("--colossum", action="store_true")
    parser.add_argument("--few_shot", action="store_true")
    
    parser.add_argument("--freeze_language_model", action="store_true")
    parser.add_argument("--freeze_vision_tower", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--add_proprio", action="store_true")
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--pretrained_rlbench_dir", type=str, default=None)
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument("--output_arm_flag", action="store_true", default=False, help="是否输出机械臂flag")
    parser.add_argument(
        "--cameras",
        type=str,  # 每个值的类型
        nargs="+",  # 接受一个或多个值
        default=["3rd"],  # 默认值
        help="List of camera names"
    )
    parser.add_argument("--update_dpo", action="store_true",default=False, help="使用DPO训练模式")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO温度参数beta")
    # parser.add_argument("--reference_model_path", type=str, default=None, help="参考模型路径，用于DPO训练")
    cmd_args = parser.parse_args()
    experiment(cmd_args)

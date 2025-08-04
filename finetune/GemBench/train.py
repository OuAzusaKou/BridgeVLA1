'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/train.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os
import subprocess
import time
import tqdm
import yaml
import argparse
import time
from contextlib import redirect_stdout
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod

from bridgevla.mvt.mvt import MVT
from bridgevla.models.bridgevla_agent import print_loss_log
from bridgevla.utils.rvt_utils import (
    get_num_feat,
)
from utils.peract_utils_gembench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
import socket
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from gembench_dataset import Gembench_Dataset

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))  # 0 will make the OS choose an available port
        return s.getsockname()[1]


def create_dataloader(dataset, rank, world_size, batch_size, num_workers):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True 
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True, 
        pin_memory=True  
    )
    return dataloader

def reduce_value(value, average=True):
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / dist.get_world_size() if average else tensor.item()


def train(agent, data_loader, cameras=["front", "left_shoulder", "right_shoulder", "wrist"],rank=0):
    agent.train()
    print(f"You are using {cameras} for training")
    def move_tensors_to_device(d, device):
        if isinstance(d, dict):
            return {k: move_tensors_to_device(v, device) if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in d.items()}
        return d
    iteration=0
    epoch_losses={}
    for raw_batch in  tqdm.tqdm(data_loader, disable=(rank != 0), position=0, leave=True):
        iteration+=1
        batch = move_tensors_to_device(raw_batch, agent._device)
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"]=[[[item]] for item in raw_batch["lang_goal"]]
        update_args = {
          "cameras":cameras,
        }
        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
            }
        )
        out=agent.update_gembench(**update_args)
        if epoch_losses=={}:
            epoch_losses = {key: [] for key in out.keys()}
        for key in epoch_losses:
            
            loss_value = out[key]
            reduced_loss = reduce_value(loss_value, average=True)
            epoch_losses[key].append(reduced_loss)



    if rank == 0:
        log = print_loss_log(agent)
    avg_losses = {key: sum(values)/len(values) for key, values in epoch_losses.items()}
    return avg_losses

def save_agent(agent, path, epoch):
    model = agent._network

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
        },
        path,
    )


def get_time():
    import datetime
    now = datetime.datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    folder_name = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return folder_name


def get_logdir(cmd_args, exp_cfg,dist):
    log_dir = os.path.join(cmd_args.log_dir,"train" ,exp_cfg.exp_id,cmd_args.exp_note)
    if cmd_args.debug==True:
        log_dir = os.path.join(log_dir,"debug")

    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    trial_time=get_time()
    log_dir = os.path.join(log_dir,f"trial_{trial_time}")
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
            # os.environ["MASTER_PORT"] = "29566"
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
        
    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id
    
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"
    

    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()


    BATCH_SIZE_TRAIN = exp_cfg.bs
    print(f"BATCH_SIZE_TRAIN={BATCH_SIZE_TRAIN}")

    


    EPOCHS = exp_cfg.epochs
    data_folder=cmd_args.data_folder
            


    log_dir = get_logdir(cmd_args, exp_cfg,dist)
    t_start = time.time()
    train_dataset = Gembench_Dataset(data_folder,device=device_id,cameras=cmd_args.cameras,ep_per_task=cmd_args.ep_per_task)
    print("Total tasks: ",train_dataset.num_tasks)
    print("Total trajectories: ",train_dataset.num_task_paths)
    print("Dataset Length: " , len(train_dataset))
    

    train_dataloader=create_dataloader(train_dataset, rank, world_size, BATCH_SIZE_TRAIN, exp_cfg.num_workers)
    
    t_end = time.time()
    if local_rank== 0:
        print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

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
        **mvt_cfg,
    )
    backbone=backbone.to(local_rank)# local rank rather than rank
    backbone = DDP(backbone, device_ids=[local_rank],find_unused_parameters=True)

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    freeze_names=["lm_head","embed_tokens"]
    if cmd_args.freeze_vision_tower:
        freeze_names.append("vision_tower")
        print("Freeze vision tower")
    for name, module in agent._network.named_modules():
        for freeze_name in freeze_names:
            if freeze_name in name:
                for param in module.parameters():
                    param.requires_grad = False
                break
    

    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)

    total_params_billion = total_params / 1e9  

    print(f'Total trainable parameters: {total_params_billion:.2f} billion')
    
    
    agent.build(training=True, device=device_id)



    start_epoch = 0
    end_epoch = EPOCHS

    if dist.get_rank() == 0:
        ## logging unchanged values to reproduce the same setting
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
        wandb.login(key="0fa0d98600c7e9cae06a14debb71ced7b8dd2a63")
        if  cmd_args.debug:
            wandb.init(entity="1113255710", project="3DVLA_RVT_opensource", name=os.path.dirname(log_dir),mode="disabled")
        else:
            wandb.init(entity="1113255710", project="3DVLA_RVT_opensource", name=os.path.dirname(log_dir))


    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training on train dataset")

        out = train(agent, train_dataloader, rank=dist.get_rank(),cameras=cmd_args.cameras)
        if rank == 0:
            wandb.log(out,step=i)
        if dist.get_rank()==0 and (i %20==0 or i == end_epoch-1):
            # TODO: add logic to only save some models
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_last.pth", i)
        i += 1
        dist.barrier()

    dist.barrier()
    if dist.get_rank() == 0:
        print("[Finish]")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--mvt_cfg_path", type=str, default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/gembench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--exp_note", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="/home/wzh/BridgeVLA/finetune/GemBench/logs")
    parser.add_argument("--data_folder", type=str, default="/PATH_TO_TRAIN_DATA/train_dataset")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ep_per_task", type=int, default=10000) # use all data
    parser.add_argument("--freeze_vision_tower", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument(
        "--cameras",
        type=str, 
        nargs="+",  
        default=["left_shoulder", "right_shoulder", "wrist", "front"], 
        help="List of camera names"
    )
    cmd_args = parser.parse_args()
    experiment(cmd_args)

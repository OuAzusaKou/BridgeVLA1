# finetune/Real/pipeline_real_dataset.py

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
from scipy.spatial.transform import Rotation as R
import gc
from real_dataset import read_action_file



class Pipeline_RealDataset(Dataset):
    def __init__(self, 
                 data_paths, 
                 device,
                 cameras=["3rd"], 
                 ep_per_task=10, 
                 output_arm_flag=False, 
                 dpo_dataset=False, 
                 current_pose_input=False,
                 add_stop_token=False):
        super().__init__()
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.cameras = cameras
        self.ep_per_task = ep_per_task
        self.output_arm_flag = output_arm_flag
        self.dpo_dataset = dpo_dataset
        self.current_pose_input = current_pose_input

        self.index_list = []
        self.all_lang_goals = set()  # DPO模式下收集所有lang_goal
        self._build_index()



    def convert_pcd_to_base(
            self, 
            extrinsic_path,
            type="3rd",
            pcd=[]
        ):
        with open(extrinsic_path, "rb") as f:
            data = pickle.load(f)
            transform = np.array(data)
        # zed相机是RGBA，所以pcd的形状为(1080, 1920, 4)
        
        h, w = pcd.shape[:2]
        pcd = pcd.reshape(-1, 3) #去掉A
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
        pcd = (transform @ pcd.T).T[:, :3]
        
        pcd = pcd.reshape(h, w, 3)
        return pcd 

    def _build_index(self):
        # 先收集所有lang_goal（DPO模式下）
        self.num_tasks = 0
        for data_path in self.data_paths:
            self.num_tasks += len([path_name for path_name in os.listdir(data_path) 
                                 if os.path.isdir(os.path.join(data_path, path_name))])


        self.num_task_paths = 0

        if self.dpo_dataset:
            for data_path in self.data_paths:
                for task in os.listdir(data_path):
                    task_path = os.path.join(data_path, task)
                    if os.path.isdir(task_path):
                        for episode_num in os.listdir(task_path):
                            if int(episode_num) >= self.ep_per_task:
                                continue
                            self.num_task_paths+=1
                            episode_path = os.path.join(task_path, episode_num)
                            instr_path = os.path.join(episode_path, "instruction.pkl")
                            if os.path.exists(instr_path):
                                with open(instr_path, 'rb') as f:
                                    instruction = pickle.load(f).strip()
                                    self.all_lang_goals.add(instruction)
            self.all_lang_goals = list(self.all_lang_goals)

        # 构建索引
        for data_path in self.data_paths:
            for task in os.listdir(data_path):
                task_path = os.path.join(data_path, task)
                if os.path.isdir(task_path):
                    for episode_num in os.listdir(task_path):
                        if int(episode_num) >= self.ep_per_task:
                            continue
                        episode_path = os.path.join(task_path, episode_num)
                        action_path = os.path.join(episode_path, 'pose.pkl')
                        rgb_3rd = os.path.join(episode_path, "zed_rgb")
                        # 读取动作文件，获取 num_steps
                        if not os.path.exists(action_path) or not os.path.exists(rgb_3rd):
                            continue
                        gripper_pose = read_action_file(action_path)
                        num_steps = sum(1 for file_name in os.listdir(rgb_3rd) if file_name.endswith('.pkl'))
                        for step in range(num_steps-1):
                            index_info = {
                                "task": task,
                                "episode_path": episode_path,
                                "step": step,
                                "episode_length":num_steps
                            }
                            if self.dpo_dataset:
                                # 记录正例lang_goal
                                instr_path = os.path.join(episode_path, "instruction.pkl")
                                if os.path.exists(instr_path):
                                    with open(instr_path, 'rb') as f:
                                        instruction = pickle.load(f).strip()
                                    index_info["lang_goal"] = instruction
                            self.index_list.append(index_info)
        gc.collect()
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        info = self.index_list[idx]
        episode_path = info["episode_path"]
        episode_length = info['episode_length']
        step = info["step"]
        task = info["task"]

        action_path = os.path.join(episode_path, 'pose.pkl')
        rgb_3rd = os.path.join(episode_path, "zed_rgb")
        pcd_3rd = os.path.join(episode_path, "zed_pcd")
        instr_path = os.path.join(episode_path, "instruction.pkl")

        gripper_pose = read_action_file(action_path)
        # 处理gripper_pose
        gripper_pose_xyz = np.array(gripper_pose[step+1]["position"]) / 1000
        gripper_pose_euler = gripper_pose[step+1]["orientation"]
        gripper_pose_quat = R.from_euler('xyz', gripper_pose_euler, degrees=True).as_quat()
        gripper_pose_vec = np.concatenate((gripper_pose_xyz, gripper_pose_quat, [gripper_pose[step+1]["claw_status"]]), axis=0)

        current_gripper_pose_xyz = np.array(gripper_pose[step]["position"]) / 1000
        current_gripper_pose_euler = gripper_pose[step]["orientation"]
        current_gripper_pose_quat = R.from_euler('xyz', current_gripper_pose_euler, degrees=True).as_quat()
        current_gripper_pose_vec = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat, [gripper_pose[step]["claw_status"]]), axis=0)

        current_gripper_state = gripper_pose[step]["claw_status"]
        num_steps = sum(1 for file_name in os.listdir(rgb_3rd) if file_name.endswith('.pkl'))
        time_val = (1. - (step / float(num_steps - 1))) * 2. - 1.
        low_dim_state = np.concatenate([[current_gripper_state], [time_val]]).astype(np.float32)

        sample = {
            "gripper_pose": gripper_pose_vec,
            "low_dim_state": low_dim_state,
            "ignore_collisions": 1.0,
            "tasks": task,
        }

        sample["3rd"], sample["wrist"] = {}, {}

        # 读取图像和点云
        if "3rd" in self.cameras:
            with open(os.path.join(rgb_3rd, f"{step}.pkl"), 'rb') as f:
                # sample["3rd"] = {"rgb": np.transpose(np.ascontiguousarray(pickle.load(f)[:, :, :3]), [2, 0, 1])}
                sample["3rd"]["rgb"] = pickle.load(f)[:, :, :3]
                # sample["3rd"]["rgb"] = joblib.load(f)[:, :, :3]
                # sample["3rd"]["rgb"] = pickle.load(f, encoding='latin1')[:, :, :3]
                sample["3rd"]["rgb"] = np.ascontiguousarray(sample["3rd"]["rgb"])  #   check it  the final image should be RGB
                sample["3rd"]["rgb"] = np.transpose(sample["3rd"]["rgb"], [2, 0, 1]) 
            with open(os.path.join(pcd_3rd, f"{step}.pkl"), 'rb') as f:
                sample["3rd"]["pcd"] = pickle.load(f)[:, :, :3]
                # sample["3rd"]["pcd"] = joblib.load(f)[:, :, :3]  
                sample["3rd"]["pcd"] = self.convert_pcd_to_base(pcd=sample["3rd"]["pcd"], type="3rd",extrinsic_path=os.path.join(episode_path, "extrinsic_matrix.pkl"))
                sample["3rd"]["pcd"] = np.transpose(sample["3rd"]["pcd"], [2, 0, 1]).astype(np.float32)

        if "wrist"  in self.cameras:
            assert False
            with open(os.path.join(rgb_wrist, f"{step}.pkl"), 'rb') as f:
                sample["wrist"]["rgb"] = pickle.load(f)
                sample["wrist"]["rgb"] = np.ascontiguousarray(sample["wrist"]["rgb"][:, :, ::-1])  # BGR转RGB（H,W,C）
                sample["wrist"]["rgb"] = np.transpose(sample["wrist"]["rgb"], [2, 0, 1])
            with open(os.path.join(pcd_wrist, f"{step}.pkl"), 'rb') as f:
                sample["wrist"]["pcd"] = pickle.load(f)
                sample["wrist"]["pcd"] = self.convert_pcd_to_base(pcd=sample["wrist"]["pcd"], type="wrist")
                sample["wrist"]["pcd"] = np.transpose(sample["wrist"]["pcd"], [2, 0, 1])
    

        # 读取指令
        with open(instr_path, 'rb') as f:
            lang_goal = pickle.load(f).strip()

        if self.output_arm_flag:
            sample["arm_flag"] = gripper_pose[step+1]["arm_flag"]

        # DPO模式
        if self.dpo_dataset:
            positive_sample = copy.deepcopy(sample)
            positive_sample["lang_goal"] = lang_goal
            negative_lang_goal = None
            available_lang_goals = [goal for goal in self.all_lang_goals if goal.split(" ")[-1] != lang_goal.split(" ")[-1]]
            if available_lang_goals:
                negative_lang_goal = np.random.choice(available_lang_goals)
            else:
                negative_lang_goal = "这是一个错误的指令"
            negative_sample = copy.deepcopy(sample)
            

            if step < episode_length - 2:

                next_gripper_pose_xyz = np.array(gripper_pose[step+2]["position"]) / 1000
                next_gripper_pose_euler = gripper_pose[step+2]["orientation"]
                next_gripper_pose_quat = R.from_euler('xyz', next_gripper_pose_euler, degrees=True).as_quat()
                next_gripper_pose_vec = np.concatenate((next_gripper_pose_xyz, next_gripper_pose_quat, [gripper_pose[step+2]["claw_status"]]), axis=0)
                negative_sample['gripper_pose'] = next_gripper_pose_vec
                negative_sample["lang_goal"] = lang_goal
            else:
                negative_sample["lang_goal"] = negative_lang_goal
            # current_pose_input
            if self.current_pose_input:
                positive_sample["current_gripper_pose"] = current_gripper_pose_vec
                negative_sample["current_gripper_pose"] = current_gripper_pose_vec
                positive_sample['lang_goal'] = lang_goal + "，当前末端姿态为:" + str(positive_sample["current_gripper_pose"])
                negative_sample['lang_goal'] = negative_lang_goal + "，当前末端姿态为:" + str(negative_sample["current_gripper_pose"])

            return {"positive": positive_sample, "negative": negative_sample}
        else:
            sample["lang_goal"] = lang_goal
            if self.current_pose_input:
                sample["current_gripper_pose"] = current_gripper_pose_vec
                sample['lang_goal'] = lang_goal + "，当前末端姿态为:" + str(sample["current_gripper_pose"])
            return sample
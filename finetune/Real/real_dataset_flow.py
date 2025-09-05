import torch
# import clip
import os
import gc
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import time
import open3d as o3d
import cv2
import random
import copy
# import joblib
from matplotlib import pyplot as plt
def read_action_file(action_path):
    '''
    文件内容类似如下
    Timestamp Position (X, Y, Z) Orientation (Rx, Ry, Rz) Claw Status
    2025-04-27_15-34-33-519 166.5982 -165.0889 168.8611 88.0599 -2.6958 -90.3270
    2025-04-27_15-34-58-985 126.8121 -441.9641 60.3471 91.3858 -4.3865 -48.2481
    2025-04-27_15-35-21-250 -53.4814 -643.2122 133.6393 90.0660 -9.3319 -89.6236
    2025-04-27_15-35-39-699 -196.8765 -642.8876 119.9657 89.8462 -8.2865 -87.1616
    2025-04-27_15-35-21-250 -53.4814 -643.2122 133.6393 90.0660 -9.3319 -89.6236
    2025-04-27_15-34-58-985 126.8121 -441.9641 60.3471 91.3858 -4.3865 -48.2481
    2025-04-27_15-34-33-519 166.5982 -165.0889 168.8611 88.0599 -2.6958 -90.3270
    '''
    with open(action_path, "rb") as f:
        data_str = pickle.load(f)
    
    # Split the string into lines
    lines = data_str['episode_data']
    
    # Initialize the result list
    result = []
    
    # Process each line of data
    for i, line in enumerate(lines):
        if i == 0:  # Skip header
            continue
            
        # Split the line into components
        
        # Extract timestamp
        timestamp = line['timestamp']
        
        # Extract position (X,Y,Z)
        position = [float(x) for x in line['end_pose'][:3]]
        
        # Extract orientation (Rx,Ry,Rz)
        orientation = [float(x) for x in line['end_pose'][3:6]]
        
        # Determine claw status based on position in sequence
        # claw_status = 1 if i % 4 == 1 or i % 4 == 0 else 0 # 1表示开，0表示关
        if len(line['end_pose']) == 9:
            claw_status = int(line['end_pose'][7])
            arm_flag = int(line['end_pose'][8])
        elif len(line['end_pose']) == 8:
            claw_status = int(line['end_pose'][7])
            arm_flag = 0
        else:
            if line['degree'] == 0:
                claw_status = 0
            else:
                claw_status = 1
            arm_flag = 0
        
        # Create dictionary for this entry
        entry = {
            'timestamp': timestamp,
            'position': position,
            'orientation': orientation,
            'claw_status': claw_status,
            'arm_flag': arm_flag,
        }
        
        result.append(entry)
    
    return result

class Real_Dataset(torch.utils.data.Dataset):
    def __init__(   
                self,
                data_path,  # 现在支持字符串或列表
                device,
                cameras,
                ep_per_task=10,
                output_arm_flag=False,
                dpo_dataset=False,
                current_pose_input=False,
                add_stop_token=False,  # 新增：是否添加截止符号
            ):
        self.device = device
        # 支持单个路径或路径列表
        if isinstance(data_path, str):
            self.data_paths = [data_path]
        else:
            self.data_paths = data_path
        self.dpo_dataset = dpo_dataset
        self.train_data = []
        self.cameras=cameras
        self.output_arm_flag = output_arm_flag
        self.current_pose_input = current_pose_input
        self.add_stop_token = add_stop_token  # 新增：截止符号标志
        self.all_lang_goals = []  # 存储所有不同的lang_goal
        print(f"You use {ep_per_task} episodes per task!")
        print(f"Data paths: {self.data_paths}")
        if self.output_arm_flag:
            print("Output arm_flag is enabled!")
        if self.dpo_dataset:
            print("DPO dataset mode is enabled!")
        if self.current_pose_input:
            print("Current pose input is enabled!")
        else:
            print("Current pose input is disabled!")
        if self.add_stop_token:
            print("Add stop token is enabled!")
        else:
            print("Add stop token is disabled!")
        time.sleep(5)
        self.construct_dataset(ep_per_task)


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

    def collect_all_lang_goals(self, ep_per_task=10):
        """收集数据集中所有不同的lang_goal"""
        print("Collecting all language goals...")
        for data_path in self.data_paths:
            for task in os.listdir(data_path):
                task_path = os.path.join(data_path, task)
                if os.path.isdir(task_path): 
                    for episode_num in os.listdir(task_path):
                        if int(episode_num) >= ep_per_task:
                            continue
                        episode_path = os.path.join(task_path, episode_num)
                        with open(os.path.join(episode_path, f"instruction.pkl"), 'rb') as f:
                            instruction = pickle.load(f)
                            lang_goal = instruction.strip()
                            if lang_goal not in self.all_lang_goals:
                                self.all_lang_goals.append(lang_goal)
        print(f"Found {len(self.all_lang_goals)} different language goals")
        
    def construct_dataset(self,ep_per_task=10):
        # 如果是DPO数据集，先收集所有不同的lang_goal
        if self.dpo_dataset:
            self.collect_all_lang_goals(ep_per_task)
        
        # 计算所有路径中的任务总数
        self.num_tasks = 0
        for data_path in self.data_paths:
            self.num_tasks += len([path_name for path_name in os.listdir(data_path) 
                                 if os.path.isdir(os.path.join(data_path, path_name))])
        
        self.num_task_paths = 0
        
        # 遍历所有数据路径
        for data_path in self.data_paths:
            for task in os.listdir(data_path):
                task_path = os.path.join(data_path, task)
                if os.path.isdir(task_path): 
                    for episode_num in tqdm(os.listdir(task_path)):
                        print('episode_num',episode_num)
                        if int(episode_num) >=ep_per_task:
                            print(f"episode num {episode_num} is larger than {ep_per_task}")
                            continue
                        # if int(episode_num) %2==0:
                        #     print(f"episode num {episode_num} is even, skip it")
                        #     continue
                        self.num_task_paths+=1
                        episode_path = os.path.join(task_path, episode_num)
                        
                        action_path = os.path.join(episode_path, 'pose.pkl')
                        rgb_3rd = os.path.join(episode_path, "image/rgb/overhead_cam_rgb")
                        pcd_3rd = os.path.join(episode_path, "image/pcd/overhead_cam_pcd")
                        gripper_pose = read_action_file(action_path)
                        
                        # 获取pcd文件列表并按文件名中的数字排序
                        pcd_files = [f for f in os.listdir(pcd_3rd) if f.endswith('.pkl')]
                        pcd_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按keypoint_后的数字排序
                        
                        num_steps = len(pcd_files)
                        
                        for step, pcd_file in enumerate(pcd_files):
                            # 从文件名提取索引，如keypoint_0169.pkl -> 169
                            current_pose_index = int(pcd_file.split('_')[-1].split('.')[0])
                            
                            # pose_index是下一个pcd文件的索引
                            if step + 1 < len(pcd_files):
                                next_pcd_file = pcd_files[step + 1]
                                pose_index = int(next_pcd_file.split('_')[-1].split('.')[0])
                            else:
                                # 如果是最后一个文件，使用当前索引
                                pose_index = current_pose_index
                            
                            # 确保pose_index在gripper_pose范围内
                            if pose_index >= len(gripper_pose):
                                print(f"Warning: pose_index {pose_index} >= len(gripper_pose) {len(gripper_pose)}, skipping")
                                continue
                                
                            sample = {}
                            # Next pose action
                           
                            # sample["gripper_pose"] = np.concatenate((gripper_pose[step+1]["position"], gripper_pose[step+1]["orientation"]), axis=0)
                            # print("before:",sample["gripper_pose"][3:7])
                            # sample["gripper_pose"][3:7] = sample["gripper_pose"][[4, 5, 6, 3]] # x y z w 作为最终的输入

                            gripper_pose_xyz=np.array(gripper_pose[pose_index]["position"])/1000 # mm -> m
                            gripper_pose_euler=gripper_pose[pose_index]["orientation"]
                            gripper_pose_quat=R.from_euler('xyz', gripper_pose_euler, degrees=True).as_quat() # check it
                            sample["gripper_pose"] = np.concatenate((gripper_pose_xyz, gripper_pose_quat,[gripper_pose[pose_index]["claw_status"]]), axis=0).astype(np.float32)
                            
                            # 对于current_gripper_pose，使用当前pcd_file对应的索引
                            current_gripper_pose_xyz=np.array(gripper_pose[current_pose_index]["position"])/1000 # mm -> m
                            current_gripper_pose_euler=gripper_pose[current_pose_index]["orientation"]
                            current_gripper_pose_quat=R.from_euler('xyz', current_gripper_pose_euler, degrees=True).as_quat() 
                            current_gripper_state = gripper_pose[current_pose_index]["claw_status"]
                            sample["current_gripper_pose"] = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat,[current_gripper_state]), axis=0).astype(np.float32)
                            
                            
                            # 添加internal_gripper_pose，包含从current_pose_index到pose_index之间的所有pose
                            chunk_size = 10  # 可以设置为类参数或固定值
                            internal_poses = []
                            
                            # 收集从current_pose_index到pose_index之间的所有pose
                            start_idx = min(current_pose_index, pose_index)
                            end_idx = max(current_pose_index, pose_index)
                            
                            for idx in range(start_idx, end_idx + 1):
                                if idx < len(gripper_pose):
                                    pose_xyz = np.array(gripper_pose[idx]["position"]) / 1000  # mm -> m
                                    pose_euler = gripper_pose[idx]["orientation"]
                                    pose_quat = R.from_euler('xyz', pose_euler, degrees=True).as_quat()
                                    pose_claw = gripper_pose[idx]["claw_status"]
                                    
                                    # 组合为7维向量 [x, y, z, qx, qy, qz, qw, claw_status]
                                    pose_vector = np.concatenate((pose_xyz, pose_quat, [pose_claw]), axis=0)
                                    internal_poses.append(pose_vector)
                            
                            # 转换为固定长度的chunk
                            if len(internal_poses) == 0:
                                # 如果没有pose，用零填充
                                sample["internal_gripper_pose"] = np.zeros((chunk_size, 8), dtype=np.float32)
                            elif len(internal_poses) < chunk_size:
                                # 如果pose数量少于chunk_size，均匀复制
                                internal_poses = np.array(internal_poses)
                                indices = np.linspace(0, len(internal_poses) - 1, chunk_size, dtype=int)
                                sample["internal_gripper_pose"] = internal_poses[indices].astype(np.float32)
                            else:
                                # 如果pose数量多于chunk_size，均匀采样
                                internal_poses = np.array(internal_poses)
                                indices = np.linspace(0, len(internal_poses) - 1, chunk_size, dtype=int)
                                sample["internal_gripper_pose"] = internal_poses[indices].astype(np.float32)
                            
                            # 确保形状为 (chunk_size, 8)
                            if sample["internal_gripper_pose"].shape != (chunk_size, 8):
                                print(f"Warning: internal_gripper_pose shape is {sample['internal_gripper_pose'].shape}, expected ({chunk_size}, 8)")
                                # 如果形状不对，重新调整
                                if sample["internal_gripper_pose"].shape[0] != chunk_size:
                                    if sample["internal_gripper_pose"].shape[0] > chunk_size:
                                        indices = np.linspace(0, sample["internal_gripper_pose"].shape[0] - 1, chunk_size, dtype=int)
                                        sample["internal_gripper_pose"] = sample["internal_gripper_pose"][indices]
                                    else:
                                        # 复制最后一个pose来填充
                                        last_pose = sample["internal_gripper_pose"][-1:] if sample["internal_gripper_pose"].shape[0] > 0 else np.zeros((1, 8))
                                        padding = np.tile(last_pose, (chunk_size - sample["internal_gripper_pose"].shape[0], 1))
                                        sample["internal_gripper_pose"] = np.vstack([sample["internal_gripper_pose"], padding])
                                
                                if sample["internal_gripper_pose"].shape[1] != 8:
                                    # 如果特征维度不对，用零填充或截断
                                    if sample["internal_gripper_pose"].shape[1] < 8:
                                        padding = np.zeros((chunk_size, 8 - sample["internal_gripper_pose"].shape[1]))
                                        sample["internal_gripper_pose"] = np.hstack([sample["internal_gripper_pose"], padding])
                                    else:
                                        sample["internal_gripper_pose"] = sample["internal_gripper_pose"][:, :8]

                            time = (1. - (step / float(num_steps - 1))) * 2. - 1.
                            sample['low_dim_state'] = np.concatenate(
                                [[current_gripper_state], [time]]).astype(np.float32)
                                
                            sample["ignore_collisions"] = 1.0
                            
                            sample["3rd"], sample["wrist"] = {}, {}
                            if "3rd" in self.cameras:
                                # 构造对应的rgb文件名
                                # rgb_file = pcd_file.replace('keypoint', 'frame').replace('pkl', 'png')
                                # with open(os.path.join(rgb_3rd, rgb_file), 'rb') as f:
                                #     # sample["3rd"]["rgb"] = pickle.load(f)[:, :, :3]
                                #     sample["3rd"]["rgb"] = plt.imread(os.path.join(rgb_3rd, rgb_file))
                                #     # sample["3rd"]["rgb"] = joblib.load(f)[:, :, :3]
                                #     # sample["3rd"]["rgb"] = pickle.load(f, encoding='latin1')[:, :, :3]
                                #     sample["3rd"]["rgb"] = np.ascontiguousarray(sample["3rd"]["rgb"])  #   check it  the final image should be RGB
                                #     sample["3rd"]["rgb"] = np.transpose(sample["3rd"]["rgb"], [2, 0, 1])  # 转为（C,H,W）
                                #     rgb_data = sample["3rd"]["rgb"]
                                #     # rgb_data = np.transpose(sample["3rd"]["rgb"], (1, 2, 0))
                                    # save_image_from_array(rgb_data, save_path='output_rgb.png')
                                with open(os.path.join(pcd_3rd, pcd_file), 'rb') as f:
                                    # sample["3rd"]["pcd"]
                                    data = pickle.load(f)
                                    sample["3rd"]["pcd"] = data['pcd']
                                    # sample["3rd"]["pcd"] = joblib.load(f)[:, :, :3]  
                                    sample["3rd"]["pcd"] = self.convert_pcd_to_base(pcd=sample["3rd"]["pcd"], type="3rd",extrinsic_path=os.path.join(episode_path, "extrinsic_matrix.pkl"))
                                    sample["3rd"]["pcd"] = np.transpose(sample["3rd"]["pcd"], [2, 0, 1]).astype(np.float32)

                                    sample["3rd"]["rgb"] = data['rgb']
                                    # sample["3rd"]["rgb"] = joblib.load(f)[:, :, :3]
                                    # sample["3rd"]["rgb"] = pickle.load(f, encoding='latin1')[:, :, :3]
                                    sample["3rd"]["rgb"] = np.ascontiguousarray(sample["3rd"]["rgb"])  #   check it  the final image should be RGB
                                    sample["3rd"]["rgb"] = np.transpose(sample["3rd"]["rgb"], [2, 0, 1])  # 转为（C,H,W）
                                
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
                            
                            

                            # import open3d as o3d
                            # def vis_pcd(pcd, rgb):

                            #     # 将点云和颜色转换为二维的形状 (N, 3)
                            #     pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
                            #     rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

                            #     # 将点云和颜色信息保存为 PLY 文件
                            #     pcd = o3d.geometry.PointCloud()
                            #     pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # 设置点云位置
                            #     pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # 设置对应的颜色
                            #     # o3d.io.write_point_cloud(save_path, pcd)
                            #     o3d.visualization.draw_geometries([pcd])
                            # test_pcd = np.concatenate((sample["3rd"]["pcd"], sample["wrist"]["pcd"]), axis=0)
                            # test_rgb = np.concatenate((sample["3rd"]["rgb"], sample["wrist"]["rgb"]), axis=0)
                            # vis_pcd(test_pcd, test_rgb)
                            


                            with open(os.path.join(episode_path, f"instruction.pkl"), 'rb') as f:
                                instruction = pickle.load(f)


                            sample["lang_goal"] = instruction.strip()
                            
                            sample["tasks"] = task
                            
                            # 如果启用output_arm_flag，则添加arm_flag到样本中
                            if self.output_arm_flag:
                                sample["arm_flag"] = gripper_pose[pose_index]["arm_flag"]
                            
                            # 如果启用add_stop_token，则添加stop_token到样本中
                            if self.add_stop_token:
                                sample["stop_token"] = 0  # 添加截止符号标志
                                
                            
                            # 如果启用current_pose_input，则添加current_gripper_pose到样本中
                            
                            if self.dpo_dataset:
                                # 为DPO数据集创建正例和多个负例
                                # 正例保持原始的lang_goal
                                positive_sample = copy.deepcopy(sample)
                                
                                # 负例：为每个其他不同的lang_goal创建一个负例
                                current_lang_goal = sample["lang_goal"]
                                available_lang_goals = [goal for goal in self.all_lang_goals if goal != current_lang_goal]
                                
                                if available_lang_goals:
                                    # 为每个可用的其他lang_goal创建一个负例
                                    for negative_lang_goal in available_lang_goals:
                                        negative_sample = copy.deepcopy(sample)
                                        negative_sample["lang_goal"] = negative_lang_goal
                                        
                                        # 创建包含正例和负例的replay_sample
                                        
                                        if self.current_pose_input:
                                            positive_sample["current_gripper_pose"] = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat,[gripper_pose[pose_index]["claw_status"]]), axis=0)
                                            negative_sample["current_gripper_pose"] = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat,[gripper_pose[pose_index]["claw_status"]]), axis=0)
                                            positive_sample['lang_goal'] = current_lang_goal + "，当前末端姿态为:" + str(positive_sample["current_gripper_pose"])
                                            negative_sample['lang_goal'] = negative_lang_goal + "，当前末端姿态为:" + str(negative_sample["current_gripper_pose"])
                                    
                                        replay_sample = {
                                            "positive": positive_sample,
                                            "negative": negative_sample
                                        }

                                        
                                        self.train_data.append(replay_sample)
                                else:
                                    # 如果没有其他lang_goal，使用一个默认的负例
                                    negative_lang_goal = "这是一个错误的指令"
                                    negative_sample = copy.deepcopy(sample)
                                    negative_sample["lang_goal"] = negative_lang_goal
                                    
                                    replay_sample = {
                                        "positive": positive_sample,
                                        "negative": negative_sample
                                    }
                                    
                                    if self.current_pose_input:
                                        positive_sample["current_gripper_pose"] = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat,[gripper_pose[pose_index]["claw_status"]]), axis=0)
                                        negative_sample["current_gripper_pose"] = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat,[gripper_pose[pose_index]["claw_status"]]), axis=0)
                                        positive_sample['lang_goal'] = current_lang_goal + "，当前末端姿态为:" + str(positive_sample["current_gripper_pose"])
                                        negative_sample['lang_goal'] = negative_lang_goal + "，当前末端姿态为:" + str(negative_sample["current_gripper_pose"])
                                    
                                    self.train_data.append(replay_sample)

                                    
                            else:
                                # 原始模式，直接添加样本

                                if self.current_pose_input:
                                    sample["current_gripper_pose"] = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat,[gripper_pose[pose_index]["claw_status"]]), axis=0)
                                    sample['lang_goal'] = sample['lang_goal'] + "，当前末端姿态为:" + str(sample["current_gripper_pose"])
                                self.train_data.append(sample)

                        if self.add_stop_token:

                            sample = {}
                            # Next pose action
                            step = num_steps - 1
                            # sample["gripper_pose"] = np.concatenate((gripper_pose[step+1]["position"], gripper_pose[step+1]["orientation"]), axis=0)
                            # print("before:",sample["gripper_pose"][3:7])
                            # sample["gripper_pose"][3:7] = sample["gripper_pose"][[4, 5, 6, 3]] # x y z w 作为最终的输入

                            gripper_pose_xyz=np.array(gripper_pose[step]["position"])/1000 # mm -> m
                            gripper_pose_euler=gripper_pose[step]["orientation"]
                            gripper_pose_quat=R.from_euler('xyz', gripper_pose_euler, degrees=True).as_quat() # check it
                            sample["gripper_pose"] = np.concatenate((gripper_pose_xyz, gripper_pose_quat,[gripper_pose[step]["claw_status"]]), axis=0)
                            
                            current_gripper_pose_xyz=np.array(gripper_pose[step]["position"])/1000 # mm -> m
                            current_gripper_pose_euler=gripper_pose[step]["orientation"]
                            current_gripper_pose_quat=R.from_euler('xyz', current_gripper_pose_euler, degrees=True).as_quat() 
                            # sample["current_gripper_pose"] = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat,[gripper_pose[step]["claw_status"]]), axis=0)


                            current_gripper_state = gripper_pose[step]["claw_status"]

                            time = (1. - (step / float(num_steps - 1))) * 2. - 1.
                            sample['low_dim_state'] = np.concatenate(
                                [[current_gripper_state], [time]]).astype(np.float32)
                                
                            sample["ignore_collisions"] = 1.0
                            
                            sample["3rd"], sample["wrist"] = {}, {}
                            if "3rd" in self.cameras:
                                with open(os.path.join(rgb_3rd, f"{step}.pkl"), 'rb') as f:
                                    sample["3rd"]["rgb"] = pickle.load(f)[:, :, :3]
                                    # sample["3rd"]["rgb"] = joblib.load(f)[:, :, :3]
                                    # sample["3rd"]["rgb"] = pickle.load(f, encoding='latin1')[:, :, :3]
                                    sample["3rd"]["rgb"] = np.ascontiguousarray(sample["3rd"]["rgb"])  #   check it  the final image should be RGB
                                    sample["3rd"]["rgb"] = np.transpose(sample["3rd"]["rgb"], [2, 0, 1])  # 转为（C,H,W）
                                    rgb_data = sample["3rd"]["rgb"]
                                    # rgb_data = np.transpose(sample["3rd"]["rgb"], (1, 2, 0))
                                    # save_image_from_array(rgb_data, save_path='output_rgb.png')
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
                            
                            # import open3d as o3d
                            # def vis_pcd(pcd, rgb):

                            #     # 将点云和颜色转换为二维的形状 (N, 3)
                            #     pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
                            #     rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

                            #     # 将点云和颜色信息保存为 PLY 文件
                            #     pcd = o3d.geometry.PointCloud()
                            #     pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # 设置点云位置
                            #     pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # 设置对应的颜色
                            #     # o3d.io.write_point_cloud(save_path, pcd)
                            #     o3d.visualization.draw_geometries([pcd])
                            # test_pcd = np.concatenate((sample["3rd"]["pcd"], sample["wrist"]["pcd"]), axis=0)
                            # test_rgb = np.concatenate((sample["3rd"]["rgb"], sample["wrist"]["rgb"]), axis=0)
                            # vis_pcd(test_pcd, test_rgb)
                            


                            with open(os.path.join(episode_path, f"instruction.pkl"), 'rb') as f:
                                instruction = pickle.load(f)


                            sample["lang_goal"] = instruction.strip()
                            
                            sample["tasks"] = task

                            if self.output_arm_flag:
                                sample["arm_flag"] = gripper_pose[step]["arm_flag"]
                            
                            # 如果启用add_stop_token，则添加stop_token到样本中
                            
                            sample["stop_token"] = 1  # 添加截止符号标志

                            if self.current_pose_input:
                                sample["current_gripper_pose"] = np.concatenate((current_gripper_pose_xyz, current_gripper_pose_quat,[gripper_pose[step]["claw_status"]]), axis=0)
                                sample['lang_goal'] = sample['lang_goal'] + "，当前末端姿态为:" + str(sample["current_gripper_pose"])
                                
                            self.train_data.append(sample)



                              
        gc.collect()
        torch.cuda.empty_cache()
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]
    
def save_pcd_with_gripper_ply(pcd, rgb, gripper_pose_xyz, save_path, gripper_radius=0.02, gripper_density=1000):
    """
    保存点云和RGB为PLY文件，并在gripper_pose_xyz处添加红色球体点（与点云合并为一个文件）。
    Args:
        pcd: numpy.ndarray, shape (3, H, W)
        rgb: numpy.ndarray, shape (3, H, W), 值范围0~255或0~1
        gripper_pose_xyz: (3,) array-like, 夹爪空间坐标
        save_path: str, ply文件保存路径
        gripper_radius: float, 球体半径
        gripper_density: int, 球体点的数量
    """
    # 1. reshape为(N, 3)
    C, H, W = pcd.shape
    pcd_flat = pcd.reshape(C, -1).T  # (N, 3)
    rgb_flat = rgb.reshape(C, -1).T  # (N, 3)
    # 2. 去除NaN
    valid_mask = ~np.isnan(pcd_flat).any(axis=1)
    pcd_valid = pcd_flat[valid_mask]
    rgb_valid = rgb_flat[valid_mask]
    # 3. 归一化颜色到0~1
    if rgb_valid.max() > 1.1:
        rgb_valid = rgb_valid / 255.0
    # 4. 生成gripper球体点
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=gripper_radius)
    sphere = sphere.sample_points_uniformly(number_of_points=gripper_density)
    sphere_points = np.asarray(sphere.points) + np.array(gripper_pose_xyz).reshape(1, 3)
    sphere_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (sphere_points.shape[0], 1))  # 红色
    # 5. 合并
    pcd_all = np.vstack([pcd_valid, sphere_points])
    rgb_all = np.vstack([rgb_valid, sphere_colors])
    # 6. 构建open3d点云
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_all)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_all)
    # 7. 保存
    o3d.io.write_point_cloud(save_path, pcd_o3d)
    print(f"点云+gripper球体已保存到: {save_path}")

def save_image_from_array(arr, save_path):
    """
    将形状为(C,H,W)的numpy数组保存为图像文件。
    Args:
        arr: numpy.ndarray, shape (C,H,W), 值范围0~255或0~1
        save_path: str, 图像保存路径
    """
    # 确保数组是(C,H,W)格式
    if arr.shape[0] == 3:
        # 转换为(H,W,C)格式
        img = np.transpose(arr, (1, 2, 0))
        # 如果值范围是0~1，则缩放到0~255
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        cv2.imwrite(save_path, img)
        print(f"图像已保存到: {save_path}")
    else:
        raise ValueError("输入数组必须是(C,H,W)格式，且C=3")

def is_bgr_image(arr):
    """
    判断一个形状为(C,H,W)的numpy数组是否为BGR格式。
    通过比较第一个通道（B）和最后一个通道（R）的平均值来判断：
    如果第一个通道的平均值大于最后一个通道的平均值，则认为是BGR格式。
    Args:
        arr: numpy.ndarray, shape (C,H,W), 值范围0~255或0~1
    Returns:
        bool: True表示BGR格式，False表示RGB格式
    """
    if arr.shape[0] != 3:
        raise ValueError("输入数组必须是(C,H,W)格式，且C=3")
    # 计算第一个通道（B）和最后一个通道（R）的平均值
    avg_b = np.mean(arr[0])
    avg_r = np.mean(arr[2])
    return avg_b > avg_r

if __name__ == "__main__":
    dataset = Real_Dataset(data_path="/mnt/data1/wuxiangnan/FiveAges_demo/3D_VLA_Data", device="cuda:0",cameras="3rd",ep_per_task=3)
    print("total samples:",len(dataset))
    for data in dataset:
        print(data.keys())
        pcd=data["3rd"]["pcd"]
        rgb=data["3rd"]["rgb"]
        # 过滤掉pcd中含有NaN的点，并同步过滤rgb
        # mask = ~np.isnan(pcd).any(axis=0)  # (H, W)
        # pcd = pcd[:, mask]
        # rgb = rgb[:, mask]
        gripper_pose_xyz=data["gripper_pose"][:3]
        save_pcd_with_gripper_ply(pcd,rgb,gripper_pose_xyz,save_path=f"/mnt/data1/3D_VLA/BridgeVLA/debug/pcd_with_gripper.ply")
        a=1
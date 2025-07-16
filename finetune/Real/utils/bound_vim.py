import open3d as o3d
import os
import pickle
import numpy as np
from transforms3d.euler import euler2mat




def get_pose(pose_data, num):
    lines = pose_data.strip().split('\n')
    third_line = lines[num]  # Extract line corresponding to current pose
    value_1, value_2, value_3, value_4, value_5, value_6 = third_line.split()[1:7]
    pose = [float(value_1), float(value_2), float(value_3), float(value_4), float(value_5), float(value_6)]
    return pose


def convert_pcd_to_base(extrinsic_matrix, pcd=[]):
    transform = extrinsic_matrix

    h, w = pcd.shape[:2]
    pcd = pcd.reshape(-1, 3)

    pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd = (transform @ pcd.T).T[:, :3]

    pcd = pcd.reshape(h, w, 3)
    return pcd


def crop_point_cloud_by_bounds(pointcloud, bounds):
    # 展开阈值
    x_min, y_min, z_min, x_max, y_max, z_max = bounds

    # 转为 numpy 数组
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # 找到在范围内的所有点
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )

    # 筛选点和颜色
    points_cropped = points[mask]
    colors_cropped = colors[mask]

    # 重新创建点云对象
    pcd_cropped = o3d.geometry.PointCloud()
    pcd_cropped.points = o3d.utility.Vector3dVector(points_cropped)
    pcd_cropped.colors = o3d.utility.Vector3dVector(colors_cropped)
    return pcd_cropped

def vis_pcd_with_end_pred(pcd, rgb, extrinsic_matrix, end_pose, pred_pose):
    # Convert point cloud coordinates
    pcd = convert_pcd_to_base(extrinsic_matrix, pcd)

    # Convert point cloud and colors to flat shapes
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    # Create point cloud object
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)
    SCENE_BOUNDS = [
        -0.6,
        -0.9,
        -0.1,
        0.4,
        0.1,
        0.6,
    ]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized

    # SCENE_BOUNDS = [
    #     0.15,
    #     -0.4,
    #     -0.05,
    #     0.85,
    #     0.6,
    #     0.6,
    # ]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized

    pointcloud = crop_point_cloud_by_bounds(pointcloud, SCENE_BOUNDS)

    # Display origin coordinate frame
    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    # Process end_pose
    end_pose = [float(x) for x in end_pose]
    pos_end = np.array(end_pose[:3]) * 0.001
    angles_deg_end = np.array(end_pose[3:])
    angles_rad_end = np.deg2rad(angles_deg_end)
    rot_mat_end = euler2mat(*angles_rad_end, axes='sxyz')
    T_end = np.eye(4)
    T_end[:3, :3] = rot_mat_end
    T_end[:3, 3] = pos_end
    axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    axis_end.transform(T_end)

    # Process pred_pose
    pred_pose = [float(x) for x in pred_pose]
    pos_pred = np.array(pred_pose[:3]) * 0.001
    angles_deg_pred = np.array(pred_pose[3:])
    angles_rad_pred = np.deg2rad(angles_deg_pred)
    rot_mat_pred = euler2mat(*angles_rad_pred, axes='sxyz')
    T_pred = np.eye(4)
    T_pred[:3, :3] = rot_mat_pred
    T_pred[:3, 3] = pos_pred
    target_axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    target_axis_end.transform(T_pred)

    # Show all content
    o3d.visualization.draw_geometries([pointcloud, axis_origin, target_axis_end])


def read_action_file(action_path):
    '''
    文件内容类似如下
    Timestamp Position (X, Y, Z) Orientation (Rx, Ry, Rz) Claw Status
    2025-04-27_15-34-33-519 166.5982 -165.0889 168.8611 88.0599 -2.6958 -90.3270 1
    2025-04-27_15-34-58-985 126.8121 -441.9641 60.3471 91.3858 -4.3865 -48.2481 1
    2025-04-27_15-35-21-250 -53.4814 -643.2122 133.6393 90.0660 -9.3319 -89.6236 0
    2025-04-27_15-35-39-699 -196.8765 -642.8876 119.9657 89.8462 -8.2865 -87.1616 0
    2025-04-27_15-35-21-250 -53.4814 -643.2122 133.6393 90.0660 -9.3319 -89.6236 1
    2025-04-27_15-34-58-985 126.8121 -441.9641 60.3471 91.3858 -4.3865 -48.2481
    2025-04-27_15-34-33-519 166.5982 -165.0889 168.8611 88.0599 -2.6958 -90.3270
    '''
    with open(action_path, "rb") as f:
        data_str = pickle.load(f)

    # Split the string into lines
    lines = data_str.strip().split('\n')

    # Initialize the result list
    result = []

    # Process each line of data
    for i, line in enumerate(lines):
        if i == 0:  # Skip header
            continue

        # Split the line into components
        parts = line.strip().split()

        # Extract timestamp
        timestamp = parts[0]

        # Extract position (X,Y,Z)
        position = [float(x) for x in parts[1:4]]

        # Extract orientation (Rx,Ry,Rz)
        orientation = [float(x) for x in parts[4:7]]

        # Determine claw status based on position in sequence
        claw_status = 1 if i % 4 == 1 or i % 4 == 0 else 0  # 1表示开，0表示关
        print(i, claw_status)

        # Create dictionary for this entry
        entry = {
            'timestamp': timestamp,
            'position': position,
            'orientation': orientation,
            'claw_status': claw_status
        }

        result.append(entry)

    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = "/home/BridgeVLA/data/20250708_2/dobot_formate_put_the_koke_can_in_the_top_shelf_7_08/1"
    # data_path = "/home/guangyun/Project/FiveAges/Data/data0"
    pcd_dir = os.path.join(data_path, "zed_pcd")
    rgb_dir = os.path.join(data_path, "zed_rgb")
    pose_path = os.path.join(data_path, "pose.pkl")
    extrinsic_matrix = os.path.join(data_path, "extrinsic_matrix.pkl")

    with open(pose_path, 'rb') as f:
        pose_data = pickle.load(f)

    read_action_file(pose_path)

    with open(extrinsic_matrix, 'rb') as f:
        extrinsic_matrix = pickle.load(f)
        extrinsic_matrix = np.array(extrinsic_matrix)

    # 显示第i组数据
    for i in range(4):
        pcd_path = os.path.join(pcd_dir, f"{i}.pkl")
        rgb_path = os.path.join(rgb_dir, f"{i}.pkl")
        pose_current = get_pose(pose_data, i+1)
        pose_next = get_pose(pose_data, i+2)
        with open(pcd_path, 'rb') as f:
            pcd_data = pickle.load(f)
        with open(rgb_path, 'rb') as f:
            rgb_data = pickle.load(f)

        pcd_data = pcd_data[:, :, :3]
        rgb_data = rgb_data[:, :, :3]
        vis_pcd_with_end_pred(pcd_data, rgb_data, extrinsic_matrix, pose_current, pose_next)




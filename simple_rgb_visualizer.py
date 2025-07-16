#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的RGB图像可视化工具
可以直接在Python环境中使用
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_and_visualize_rgb(data_path, step=0, camera_type="3rd", save_path=None):
    """
    加载并可视化RGB图像
    
    Args:
        data_path: 数据路径
        step: 时间步
        camera_type: 相机类型 ("3rd" 或 "wrist")
        save_path: 保存路径（可选）
    
    Returns:
        rgb_array: RGB图像数组
    """
    # 构建文件路径
    rgb_path = os.path.join(data_path, f"rgb_{camera_type}")
    rgb_file = os.path.join(rgb_path, f"{step}.pkl")
    
    print(f"正在加载: {rgb_file}")
    
    if not os.path.exists(rgb_file):
        print(f"错误: 文件不存在 - {rgb_file}")
        return None
    
    # 加载数据
    with open(rgb_file, 'rb') as f:
        rgb_data = pickle.load(f)
    
    print(f"原始数据形状: {rgb_data.shape}")
    print(f"数据类型: {rgb_data.dtype}")
    print(f"值范围: [{rgb_data.min():.2f}, {rgb_data.max():.2f}]")
    
    # 确保数据格式正确 (C, H, W)
    if len(rgb_data.shape) == 3:
        if rgb_data.shape[2] == 3:  # (H, W, C) -> (C, H, W)
            rgb_data = np.transpose(rgb_data, (2, 0, 1))
            print("已转换为 (C, H, W) 格式")
        elif rgb_data.shape[0] == 3:  # 已经是 (C, H, W)
            print("数据已经是 (C, H, W) 格式")
        else:
            print(f"警告: 不支持的RGB数据形状: {rgb_data.shape}")
    
    # 可视化
    visualize_rgb(rgb_data, f"{camera_type.upper()} 相机 - 时间步 {step}", save_path)
    
    return rgb_data

def visualize_rgb(rgb_array, title="RGB图像", save_path=None):
    """
    可视化RGB图像
    
    Args:
        rgb_array: RGB图像数组 (C, H, W)
        title: 图像标题
        save_path: 保存路径
    """
    # 转换为 (H, W, C) 格式用于显示
    if rgb_array.shape[0] == 3:
        img_display = np.transpose(rgb_array, (1, 2, 0))
    else:
        img_display = rgb_array
    
    # 确保值范围在0-255
    if img_display.max() <= 1.0:
        img_display = (img_display * 255).astype(np.uint8)
        print("已将值范围从0-1缩放到0-255")
    else:
        img_display = img_display.astype(np.uint8)
    
    # 创建图像显示
    plt.figure(figsize=(12, 8))
    plt.imshow(img_display)
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    # 添加图像信息
    info_text = f"形状: {rgb_array.shape}\n数据类型: {rgb_array.dtype}\n值范围: [{rgb_array.min():.2f}, {rgb_array.max():.2f}]"
    plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
        
        # 同时保存为原始图像文件
        raw_save_path = save_path.replace('.png', '_raw.jpg')
        cv2.imwrite(raw_save_path, img_display)
        print(f"原始图像已保存到: {raw_save_path}")
    
    plt.show()

def explore_data_directory(data_path):
    """
    探索数据目录结构
    
    Args:
        data_path: 数据路径
    """
    print(f"探索数据目录: {data_path}")
    print("=" * 50)
    
    if not os.path.exists(data_path):
        print(f"错误: 目录不存在 - {data_path}")
        return
    
    # 列出目录内容
    try:
        items = os.listdir(data_path)
        print("目录内容:")
        for item in sorted(items):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # 如果是相机目录，显示其中的文件
                if item.startswith("rgb_") or item.startswith("pcd_"):
                    try:
                        files = os.listdir(item_path)
                        print(f"     文件数量: {len(files)}")
                        if files:
                            print(f"     示例文件: {files[0]}")
                    except:
                        pass
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"读取目录时出错: {e}")

def find_available_steps(data_path, camera_type="3rd"):
    """
    查找可用的时间步
    
    Args:
        data_path: 数据路径
        camera_type: 相机类型
    
    Returns:
        steps: 可用时间步列表
    """
    rgb_path = os.path.join(data_path, f"rgb_{camera_type}")
    
    if not os.path.exists(rgb_path):
        print(f"错误: 相机目录不存在 - {rgb_path}")
        return []
    
    try:
        files = os.listdir(rgb_path)
        # 提取时间步数字
        steps = []
        for file in files:
            if file.endswith('.pkl'):
                try:
                    step = int(file.replace('.pkl', ''))
                    steps.append(step)
                except:
                    pass
        
        steps.sort()
        return steps
    except Exception as e:
        print(f"读取时间步时出错: {e}")
        return []

# 示例使用函数
def example_usage():
    """
    示例使用方法
    """
    print("RGB图像可视化工具 - 示例用法")
    print("=" * 50)
    
    # 请替换为您的实际数据路径
    data_path = "/path/to/your/data"  # 请修改为实际路径
    
    print("1. 探索数据目录:")
    print("   explore_data_directory(data_path)")
    
    print("\n2. 查找可用时间步:")
    print("   steps = find_available_steps(data_path, '3rd')")
    print("   print(f'可用时间步: {steps}')")
    
    print("\n3. 可视化单个时间步:")
    print("   rgb_data = load_and_visualize_rgb(data_path, step=0, camera_type='3rd')")
    
    print("\n4. 保存图像:")
    print("   load_and_visualize_rgb(data_path, step=0, camera_type='3rd', save_path='output.png')")
    
    print("\n5. 可视化多个时间步:")
    print("   for step in [0, 5, 10, 15]:")
    print("       load_and_visualize_rgb(data_path, step, '3rd')")

if __name__ == "__main__":
    example_usage() 
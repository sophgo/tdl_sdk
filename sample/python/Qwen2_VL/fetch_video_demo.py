#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import json

# 添加tdl模块到Python路径
sys.path.append(os.path.abspath("../../build/lib"))
import tdl


def extract_vision_info(conversations):
    """从对话中提取视觉信息
    
    Args:
        conversations: 对话列表，可以是字典列表或嵌套的字典列表
        
    Returns:
        视觉信息列表
    """
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos

def process_vision_info(
    conversations,
    return_video_kwargs=False,
):
    """处理对话中的视觉信息
    
    Args:
        conversations: 对话列表
        return_video_kwargs: 是否返回视频参数
        
    Returns:
        处理后的图像、视频和可选的视频参数
    """
    vision_infos = extract_vision_info(conversations)
    
    # 读取图像或视频
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            img = tdl.llm.fetch_image(vision_info["image"], vision_info)
            if img:
                image_inputs.append(img)
        elif "video" in vision_info:
            video_input, video_sample_fps = tdl.llm.fetch_video(vision_info["video"], vision_info["fps"], vision_info["frames"],0)
            if video_input is not None:
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
        else:
            print("警告：视觉内容中应包含 image, image_url 或 video")
    
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs

def process_vision_demo(json_file, output_dir="./demo_output"):
    """处理多模态对话示例
    
    Args:
        json_file: 包含对话的JSON文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载示例对话
    with open(json_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # 处理视觉信息
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        conversations, return_video_kwargs=True
    )
    
    # 显示处理结果
    print("处理结果:")
    print(f"- 图像数量: {len(image_inputs) if image_inputs else 0}")
    print(f"- 视频数量: {len(video_inputs) if video_inputs else 0}")
    

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='视频帧提取与多模态对话处理示例')
    parser.add_argument('--video', type=str, help='视频文件路径')
    parser.add_argument('--frames', type=int, default=0, help='要提取的帧数')
    parser.add_argument('--fps', type=float, default=2.0, help='期望的帧率')
    args = parser.parse_args()
    

    # 基本视频处理
    if not args.video:
        print("错误：未指定视频文件。请使用 --video 参数指定视频文件，或使用 --json 参数指定对话文件。")
        return
    
    # 提取视频帧
    print(f"从视频中提取帧: {args.video}")
    vision_info = {
        "video": args.video,
        "frames": args.frames,
        "fps": args.fps
    }
    print(f"vision_info: {vision_info}")
    frames = tdl.llm.fetch_video(vision_info["video"], vision_info["fps"], vision_info["frames"],0)
    print(f"frames: {frames}")
    # 检查是否成功提取帧
    if frames is None or frames.size == 0:
        print("警告：未能提取到任何帧！")
        return
    
    # 显示结果信息
    print(f"提取了 {frames.shape[0]} 帧")
    print(f"帧尺寸: {frames.shape[1:]} (高度, 宽度, 通道)")
    print(f"数值范围: [{frames.min():.4f}, {frames.max():.4f}]")
    

if __name__ == "__main__":
    main()
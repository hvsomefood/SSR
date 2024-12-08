import pandas as pd
import os
import cv2
import numpy as np

def load_videos_from_csv(csv_file):
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    
    videos = []
    labels = []

    for index, row in data.iterrows():
        video_path = row[0]  # 假设CSV的第一列是视频路径
        label = os.path.basename(os.path.dirname(video_path))  # 获取文件夹名作为标签
        
        # 加载视频并提取帧
        frames = load_video_frames(video_path)
        
        if frames is not None:
            videos.append(frames)
            labels.append(label)
    
    return np.array(videos), np.array(labels)

def load_video_frames(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < 5:  # 取前5帧
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有读取到帧，退出循环
        
        # 将帧调整为112x112大小，并转换为灰度图
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)

    cap.release()
    
    # 检查是否有5帧
    if len(frames) == 5:
        return np.array(frames)
    else:
        return None  # 如果帧数不足，返回None

if __name__ == "__main__":
    train_videos, train_labels = load_videos_from_csv('D:/SSR/CMLRdataset/Processed/train/training_csv.csv')
    val_videos, val_labels = load_videos_from_csv('D:/SSR/CMLRdataset/Processed/val/validation_csv.csv')
    test_videos, test_labels = load_videos_from_csv('D:/SSR/CMLRdataset/Processed/test/testing_csv.csv')


def count_letters(input_array):
    # 创建一个字典用于统计字母
    letter_count = {}
    
    # 遍历输入数组
    for letter in input_array:
        if letter.isupper():  # 确保是大写字母
            if letter in letter_count:
                letter_count[letter] += 1
            else:
                letter_count[letter] = 1

    # 统计种类和总数
    n = len(letter_count)  # 不同字母的种类数
    m = sum(letter_count.values())  # 所有字母的总数

    return n, letter_count, m



def encode_letters(input_array):
    # 创建一个按字母排序的列表
    sorted_letters = sorted(set(input_array))  # 获取唯一字母并排序
    
    # 创建一个字典用于映射字母到编码
    letter_to_index = {letter: index for index, letter in enumerate(sorted_letters)}

    # 编码原始数组
    encoded_array = [letter_to_index[letter] for letter in input_array if letter in letter_to_index]
    
    return encoded_array, letter_to_index
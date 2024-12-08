from IPython.display import HTML, Video, display
from moviepy.editor import VideoFileClip
import os
from playsound import playsound
from base64 import b64encode
from pypinyin import pinyin, lazy_pinyin
import wave
import numpy as np
import subprocess
import cv2
import mediapipe as mp
import random
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

PROJECT_FOLDER = 'D:/SSR/'
DATASET_FOLDER = PROJECT_FOLDER + 'CMLRdataset/'
VIDEO_FOLDER = DATASET_FOLDER + 'video/'
AUDIO_FOLDER = DATASET_FOLDER + 'audio/'
TEXT_FOLDER = DATASET_FOLDER + 'text/'


def get_video_file(file_path, play=False):

    video_path = get_video_path(file_path)
    video_clip = VideoFileClip(video_path)
    
    if play:
        duration = video_clip.duration
        display(Video(file_path, embed=True, width=480, height=360))
        print(f"视频路径: {video_path}")
        print(f"视频的时长为: {duration:.2f} 秒")
    return video_clip


def get_audio_file(file_path, play=True):

    audio_path = get_audio_path(file_path)
    with wave.open(audio_path, 'rb') as wav_file:
        # 获取参数
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # 读取音频数据
        frames = wav_file.readframes(n_frames)
        
        # 将字节数据转换为 NumPy 数组
        audio_data = np.frombuffer(frames, dtype=np.int16)  # 假设音频是 16-bit PCM
        audio_data = audio_data.reshape(-1, n_channels)  # 如果是多通道，调整形状
    if play:
        playsound(audio_path)
        print(f"音频文件路径: {audio_path}")
    return audio_data
    

def get_text_file(file_path, print_flag=False):
    text_path = get_text_path(file_path)
    with open(text_path, 'r', encoding='utf-8') as file:
        content = file.read()
    if print_flag:
        print(f"文字文件路径: {text_path}")
        print(content)
    return(content)


def get_video_path(file_path):

    video_file_name = f"{file_path}.mp4"  # 添加文件扩展名
    video_path = os.path.join(VIDEO_FOLDER, video_file_name)
    
    return video_path


def get_audio_path(file_path):
 
    audio_file_name = f"{file_path}.wav"  # 添加文件扩展名
    audio_relative_path_path = os.path.join(AUDIO_FOLDER, audio_file_name)
    
    return audio_relative_path_path


def get_text_path(file_path):
 
    text_file_name = f"{file_path}.txt"  # 添加文件扩展名
    text_path = os.path.join(TEXT_FOLDER, text_file_name)

    return(text_path)
    
def process_txt_file(file_path):
    """
    处理给定的文本文件，返回包含起止时间、字词和声母的数组。
    
    :param file_path: 文本文件的路径
    :return: 包含每个字词的起止时间和声母的列表
    """
    text_path = get_text_path(file_path)
    result = []
    
    with open(text_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # 跳过格式不正确的行
            
            start_time = float(parts[0])
            end_time = float(parts[1])
            word = ''.join(parts[2:])  # 将字词连接起来
            
            # 提取声母（首字母大写）
            initials = ''.join([p[0].upper() for p in lazy_pinyin(word)])  # 只提取声母并转换为大写
            if len(initials) == 0:
                initials = '_'  # 如果没有声母，使用下划线作为占位符
            
            result.append((index, start_time, end_time, initials))
    
    return result
    
def extract_initials_from_text_file(file_path):
    """
    从文本文件中提取声母并生成一个字符串。
    
    :param file_path: 文本文件的路径
    :return: 由声母组成的字符串
    """

    text_path = get_text_path(file_path)
    with open(text_path, 'r', encoding='utf-8') as file:
        # 读取第一行，作为字幕
        subtitle = file.readline().strip()
        
        # 提取声母
        initials = ''.join([p[0].upper() for p in lazy_pinyin(subtitle)])

    return initials


def round_to_frame_time(time, fps):
    """将时间四舍五入到最近的帧时间."""
    return round(time * fps) / fps


def adjust_to_nearest_multiple(value, multiple):
    """调整值到最近的倍数."""
    return round(value / multiple) * multiple


def cut_video(input_video_path, segments, output_dir):
    """
    根据给定的时间段裁切视频并保存到指定位置。
    
    :param input_video_path: 输入视频的路径
    :param segments: [(index, start_time, end_time, initials)] 格式的列表
    :param output_dir: 输出视频保存的目录
    """

    video_path = get_video_path(input_video_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录

    fps = 25  # 视频帧率
    frame_duration = 1 / fps  # 每帧的持续时间

    for index, start_time, end_time, initials in segments:
        # 将起止时间总共减去 0.15 秒，但确保起始时间不小于 0
        start_time = max(0, start_time - 0.1)
        
         # 使用 adjust_to_nearest_multiple 调整起始时间
        start_time = adjust_to_nearest_multiple(start_time, frame_duration)

        num_parts = len(initials)  # 字数或词数
        
        if num_parts == 1:  # 单个字
        
        # 读取当前文件夹下的文件数量
            output_folder = os.path.join(output_dir, initials)
            current_data_num = len([f for f in os.listdir(output_folder) if f.endswith('.mp4')])
            output_file = os.path.join(output_folder, f"{current_data_num + 1}.mp4")
            command = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(0.2), # duration = 0.2
                '-c:v', 'libx264',  # 使用H.264编码视频
                '-c:a', 'aac',      # 使用AAC编码音频
                output_file
            ]
            #print(f"Executing command: {' '.join(command)}")  # 打印命令
            result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Error: {result.stderr.decode()}")  # 打印错误信息

        else:  # 多个字或词组
            
            for i in range(num_parts):
                
                output_folder = os.path.join(output_dir, initials[i])
                current_data_num = len([f for f in os.listdir(output_folder) if f.endswith('.mp4')])
                output_file = os.path.join(output_folder, f"{current_data_num + 1}.mp4")

                segment_start = start_time + i * 0.2
                command = [
                    'ffmpeg', '-i', video_path,
                    '-ss', str(segment_start),
                    '-t', str(0.2),
                    '-c:v', 'libx264',  # 使用H.264编码视频
                    '-c:a', 'aac',      # 使用AAC编码音频
                    output_file
                ]
                #print(f"Executing command: {' '.join(command)}")  # 打印命令
                result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                if result.returncode != 0:
                    print(f"Error: {result.stderr.decode()}")  # 打印错误信息


def get_video_frames(video_path):
    """提取视频的所有帧并返回帧列表。"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"Warning: No frames extracted from video: {video_path}")
    return frames

def adjust_frame_count(frames, target_count=5):
    """调整帧数为指定的数量，使用复制和裁剪。"""
    if not frames:
        print("Error: Frame list is empty. Returning an empty list.")
        return frames  # 或者返回默认值
    current_count = len(frames)
    if current_count < target_count:
        # 复制最后一帧补足
        while len(frames) < target_count:
            frames.append(frames[-1])
    elif current_count > target_count:
        # 裁剪多余的帧
        frames = frames[:target_count]
    return frames

def extract_keypoints(frame):
    """使用 Mediapipe 提取嘴唇的关键点位置，返回嘴唇关键点的坐标。"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    key_number = [61, 105, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146] 
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 提取新的嘴唇关键点
            lip_keypoints = np.array([[landmarks[i].x, landmarks[i].y] for i in key_number])

            return lip_keypoints
        else:
            print("No face landmarks detected.")
            return np.zeros((len(key_number), 2))  # 返回与 key_number 对应数量的零坐标

def calculate_mean_keypoints(keypoints):
    """计算关键点的均值。"""
    return round(np.mean(keypoints, axis=0))

def crop_to_lips(frame, lip_keypoints, size=(50, 35)):
    """根据嘴唇关键点裁剪图像，返回指定大小的图像。"""
    # 计算嘴唇的中心位置
    center_x = np.mean(lip_keypoints[:, 0]) * frame.shape[1]
    center_y = np.mean(lip_keypoints[:, 1]) * frame.shape[0] + 5
    center_x, center_y = int(center_x), int(center_y)

    half_width, half_height = size[0] // 2, size[1] // 2
    x1 = max(center_x - half_width, 0)
    y1 = max(center_y - half_height, 0)
    x2 = min(center_x + half_width, frame.shape[1])
    y2 = min(center_y + half_height, frame.shape[0])

    cropped_frame = frame[y1:y2, x1:x2]
    
    # Resize to ensure output size is (112, 112)
    if cropped_frame.shape[0] != size[1] or cropped_frame.shape[1] != size[0]:
        cropped_frame = cv2.resize(cropped_frame, size)

    return cropped_frame


def resize_and_pad(frame, target_size=(112, 112)):
    """将图像等比例放大到目标大小，并用零填充剩余部分。"""
    h, w = frame.shape[:2]
    target_h, target_w = target_size

    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    
    # 计算新的尺寸
    new_size = (int(w * scale), int(h * scale))
    
    # 调整图像大小
    resized_frame = cv2.resize(frame, new_size)

    # 创建目标大小的零填充图像
    padded_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 计算填充的位置
    x_offset = (target_w - new_size[0]) // 2
    y_offset = (target_h - new_size[1]) // 2

    # 将调整后的图像放入填充图像中
    padded_frame[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized_frame

    return padded_frame


def process_video(video_path):

    """处理视频并返回处理后的5*112*112的数组。"""
    frames = get_video_frames(video_path)
    if not frames:
        print(f"No frames available for processing: {video_path}")
        return
    frames = adjust_frame_count(frames)

    keypoints_list = []
    processed_frames = []
    
    for frame in frames:
        lip_keypoints = extract_keypoints(frame)
        keypoints_list.append(lip_keypoints)

        # 根据嘴唇关键点裁剪图像
        cropped_frame = crop_to_lips(frame, lip_keypoints)
        
        # 调整并填充图像
        padded_frame = resize_and_pad(cropped_frame, target_size=(112, 112))
        
        gray_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2GRAY)
        processed_frames.append(gray_frame)

    # 转换为 NumPy 数组并调整形状
    processed_array = np.array(processed_frames)

    # 保存处理后的结果覆盖原始视频
    height, width = processed_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 25, (width, height), isColor=False)

    for frame in processed_frames:
        out.write(frame)
    out.release()

    return processed_array

def batch_processing(test_data, max_process_num, output_path):
    # 计数器初始化
    processed_count = 0

    for i in range(len(test_data)):
        # 获取当前视频路径
        video_path = test_data.iloc[i, 0]

        # 检查视频路径是否以 s1 到 s6 开头
        if video_path.startswith(('s1', 's2', 's3', 's4', 's5', 's6')):
            # 获取视频切割信息
            result = process_txt_file(video_path)  # 假设这个函数返回您提供的格式

            # 调用 cut_video 函数剪裁视频
            cut_video(video_path, result, output_path)

            # 累计处理的视频数量
            processed_count += 1

            print(f"Processed video: {video_path}")

            # 检查是否达到最大处理数量
            if processed_count >= max_process_num:
                print("Reached maximum number of processed videos.")
                break
        else:
            print(f"Skipped video: {video_path} (not in s1 to s6)")


    # 获取从 A 到 Z 的文件夹
    folders = [folder for folder in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, folder))]
    # 遍历每个文件夹
    for folder in folders:
        folder_path = os.path.join(output_path, folder)
        
        # 检查文件夹是否存在
        if os.path.exists(folder_path):
            # 遍历文件夹中的所有 MP4 文件
            for filename in os.listdir(folder_path):
                if filename.endswith('.mp4'):
                    file_path = os.path.join(folder_path, filename)
                    
                    # 处理视频
                    process_video(file_path)
                    
                    print(f"Processed video: {file_path}")
        else:
            print(f"Folder not found: {folder_path}")

def csv(csv_name, datafile):
    # 存储所有找到的 mp4 视频路径
    video_paths = []

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(datafile):
        for file in files:
            if file.endswith('.mp4'):
                # 获取完整视频路径
                full_path = os.path.join(root, file)
                video_paths.append(full_path)

    # 打乱视频路径
    random.shuffle(video_paths)

    # 将打乱后的路径转换为 DataFrame
    df = pd.DataFrame(video_paths, columns=['视频路径'])

    # 指定输出 CSV 文件的路径
    output_csv_path = os.path.join('C:/Users/fyh14/OneDrive/Desktop', csv_name)

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"CSV 文件已保存至: {output_csv_path}")



def validate_csv(input_csv, output_csv):
    # 读取 CSV 文件
    df = pd.read_csv(input_csv)

    # 检查'视频路径'列是否存在
    if '视频路径' not in df.columns:
        print("Error: '视频路径' 列在 CSV 文件中不存在。")
        return

    # 验证路径的存在性
    valid_paths = []
    for path in df['视频路径']:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"路径不存在，已删除: {path}")

    # 创建新的 DataFrame 仅包含有效路径
    valid_df = pd.DataFrame(valid_paths, columns=['视频路径'])

    # 保存有效路径到新的 CSV 文件
    valid_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"有效路径已保存至: {output_csv}")

def check_data_format(videos, labels):
    # 检查视频数据形状
    print("Video data shape:", videos.shape)  # 应该是 (6681, 5, 112, 112, 1)
    print("Label data shape:", labels.shape)   # 应该是 (6681,)

    # 检查数据类型
    print("Video data type:", videos.dtype)     # 应该是 torch.float32
    print("Label data type:", labels.dtype)      # 应该是 torch.int64

    # 检查每个视频和标签的样本数是否匹配
    assert videos.shape[0] == labels.shape[0], "Number of videos and labels must match!"
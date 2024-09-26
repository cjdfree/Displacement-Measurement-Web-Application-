import os
import traceback

import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
from PIL import Image
import json


class ExperimentYOLO:
    def __init__(self, video_filename, video_path, selected_model, start_time, end_time, socketio):
        self.result_root_path = r'./ultralytics/results'
        self.video_filename = os.path.splitext(os.path.basename(video_filename))[0]  # 仅截取前面的文件名，不包含扩展名: normal
        self.video_path = video_path
        self.selected_model = os.path.splitext(os.path.basename(selected_model))[0]  # 仅截取前面的文件名，不包含扩展名: yolov8n
        self.start_time = start_time
        self.end_time = end_time
        self.socketio = socketio
        self.output_path = None  # 将视频裁剪后放到的文件夹的位置

    def control(self):
        try:
            self.make_results_dir()  # 创建项目文件夹
            self.extract_frames()  # 把上传的视频保存成图像帧
            self.test_save_roi()  # yolo预测，保存识别目标的ROI和annotated_images
            self.create_video_from_images()  # 将annotated_images生成视频

        except Exception as e:
            print("detect & displacement Failure!!!")
            print(e)
            traceback.print_exc()

    def make_results_dir(self):
        os.makedirs(self.result_root_path, exist_ok=True)  # 结果根目录: r'./ultralytics/results'
        os.makedirs(os.path.join(self.result_root_path, self.video_filename), exist_ok=True)  # 根据上传视频的名字建立文件夹
        os.makedirs(os.path.join(self.result_root_path, self.video_filename, self.selected_model), exist_ok=True)  # 根据选择的模型建立文件夹
        os.makedirs(os.path.join(self.result_root_path, self.video_filename, self.selected_model, 'annotated_images'), exist_ok=True)
        os.makedirs(os.path.join(self.result_root_path, self.video_filename, self.selected_model, 'calibration_roi'), exist_ok=True)

    def extract_frames(self):
        self.output_path = f'./static/uploads/{self.video_filename}'
        os.makedirs(f'./static/uploads/{self.video_filename}', exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)
        # 获取视频的帧率
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 计算起始和结束帧
        start_frame = int(self.start_time) * fps
        end_frame = int(self.end_time) * fps
        current_frame = 0  # 当前帧数
        saved_frame_count = 0  # 计数已保存的帧数

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame >= start_frame and current_frame <= end_frame:
                # 保存帧
                frame_filename = os.path.join(self.output_path, f'{saved_frame_count:04d}.jpg')
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1  # 增加已保存帧数

            current_frame += 1

            # 如果到达结束帧，退出循环
            if current_frame > end_frame:
                break

        # 释放视频捕获对象
        cap.release()
        print(f"Frames extracted and saved to {self.output_path}")

    def test_save_roi(self):
        # Load a model
        model = YOLO(f"./ultralytics/weights/original/{self.selected_model}.pt")  # load an official model
        model = YOLO(f"./ultralytics/weights/finetuned/{self.selected_model}.pt")  # load a custom model

        # 测试结果
        results = model(source=self.output_path, imgsz=(640, 360), half=True, iou=0.7, device=0, stream=True)
        yolo_roi = {}

        # 获取视频帧数
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release() # 释放VideoCapture对象

        for i, result in enumerate(results):
            # 保存annotated_images
            im_bgr = result.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            image_filename = os.path.join(self.result_root_path, self.video_filename, self.selected_model, 'annotated_images', f'{i:04d}.jpg')
            result.save(filename=image_filename)  # 保存结果
            self.socketio.emit('progress', {'current': i + 1, 'total': frame_count})

            # 保存yolo_roi
            yolo_roi[i] = {}
            boxes = result.boxes.xyxy.cpu().numpy()  # 坐标，转换成numpy类型
            labels = result.boxes.cls.cpu().numpy()  # 标签

            # 对前五位的Calibration进行储存
            calibration_indices = np.where(labels == 0)[0]
            calibration_boxes = boxes[calibration_indices]

            print(f'calibration_indices: {calibration_indices}')

            # 遍历calibration_boxes，取出每一层的坐标，发现不同场景不是都是下面的逻辑判断，干脆直接用从小到大排序，然后按顺序赋值
            # 按照每个子列表的第四个元素（索引为3）进行排序
            sorted_calibration_boxes = sorted(calibration_boxes, key=lambda x: x[3], reverse=True)
            for j in range(5):
                yolo_roi[i][j+1] = sorted_calibration_boxes[j].tolist()
            print(f'yolo_roi[i]: {yolo_roi[i]}')

        # 将字典写入JSON文件
        filename = os.path.join(self.result_root_path, self.video_filename, self.selected_model, 'calibration_roi', f'{self.video_filename}_{self.selected_model}_calibration.json')
        with open(filename, 'w', encoding='utf-8') as f:
            # 确保中文能够被正确编码
            json.dump(yolo_roi, f, ensure_ascii=False, indent=4)


    # 将所有的图片组成视频
    def create_video_from_images(self, fps=100):
        # 获取文件夹内所有图片文件名
        image_folder = os.path.join(self.result_root_path, self.video_filename, self.selected_model,
                                      'annotated_images')
        images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]
        print(images)

        video_filename = os.path.join(self.result_root_path, self.video_filename, self.selected_model,
                                      'annotated_video.mp4')

        # 读取第一张图片以获取尺寸
        first_image = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, _ = first_image.shape

        # 定义视频编码方式和输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        # 遍历所有图片并写入视频
        for image in images:
            img_path = os.path.join(image_folder, image)
            img = cv2.imread(img_path)
            video_writer.write(img)

        # 释放视频写入对象
        video_writer.release()
        print(f"视频已保存到 {video_filename}")

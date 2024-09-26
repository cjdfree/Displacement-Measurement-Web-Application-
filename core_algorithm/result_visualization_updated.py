import argparse
import json
import os
import shutil
import pandas as pd
from scipy.signal import resample

import numpy as np
import cv2
from matplotlib import pyplot as plt

import traceback
from function_decorator import print_function_name_decorator


class Experiment:

    @print_function_name_decorator
    def __init__(self, total_scenes, yolo_model_series, model_list, flag):

        self.total_scenes = total_scenes  # 所有数据集的场景
        self.yolo_model_series = yolo_model_series
        self.model_list = model_list  # 使用的模型
        self.flag = flag  # 帧是否更新 []

        self.file_types = ['original_pretrained_test']

        # 生成数据集和模型训练的配置参数: {'pre_trained_test': args}
        self.config = {}

        # {'steel_ruler': {'Industrial_camera_0%': path, 'Industrial_camera_5%': path},
        #  'five_floors_frameworks': {'0-4_0%': path, '01-02_5%': path}}
        # 按照上面的格式来存路径
        self.original_dataset_dict = {}  # 原始数据集的字典，里面存放路径
        self.original_pretrained_test = {}  # 存放直接预测的结果

    @print_function_name_decorator
    def init_experiment(self):
        for model_name in self.model_list:
            self.make_dir(model_name)
            for file_type in self.file_types:
                # 创建属性的字典
                experiment_attribute = getattr(self, file_type)
                experiment_attribute[model_name] = {}
                for data_type, scenes in self.total_scenes.items():
                    experiment_attribute[model_name][data_type] = {}
                    for scene in scenes:
                        experiment_attribute[model_name][data_type][scene] = {}
                        for flag in self.flag:
                            experiment_attribute[model_name][data_type][scene][flag] = \
                                os.path.join(model_name, file_type, data_type, scene, flag)

                            # 创建文件夹
                            self.make_dir(os.path.join(model_name, file_type))
                            self.make_dir(os.path.join(model_name, file_type, data_type))
                            self.make_dir(os.path.join(model_name, file_type, data_type, scene))
                            self.make_dir(os.path.join(model_name, file_type, data_type, scene, flag))

        for data_type, scenes in self.total_scenes.items():
            self.original_dataset_dict[data_type] = {}
            for scene in scenes:
                # 初始化字典，其中original_dataset_dict比较特殊，不同模型之间用的是一样的，所以没有model_name这个key
                self.original_dataset_dict[data_type][scene] = os.path.join(f'original_dataset', data_type, scene)

    @print_function_name_decorator
    def control(self):
        try:
            self.init_experiment()
            self.result_visualization()

        except Exception as e:
            print("RAFT & yolo Failure!!!")
            print(e)
            traceback.print_exc()

    @print_function_name_decorator
    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)  # 如果文件路径不存在，则创建它
            print(f"文件路径 {path} 不存在，已成功创建")
        else:
            print(f"文件路径 {path} 已存在")

    @print_function_name_decorator
    def result_visualization(self):
        for flag in self.flag:
            for model_name in self.model_list:
                for data_type, scenes in self.total_scenes.items():
                    for scene in scenes:
                        # 可视化roi选取的光流图位置
                        self.visualize_roi(flag, model_name, data_type, scene)
                        # 计算缩放因子calc_factor
                        scale_factor = self.calc_factor(flag, model_name, data_type, scene)
                        # 与excel比较compare_excel
                        self.compare_excel(flag, model_name, data_type, scene, scale_factor)

    @print_function_name_decorator
    def visualize_roi(self, flag, model_name, data_type, scene):
        # 加载对应的json文件，即已经识别好的ROI，先使用yolov8n的结果
        filename = os.path.join(r'calibration_roi\540_960', f'{scene}_yolov8_m_calibration.json')
        with open(filename, 'r', encoding='utf-8') as f:
            calibration_roi = json.load(f)
            f.close()

        # 遍历图片，画矩形框
        flow_image_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'image')
        roi_flow_image_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'roi_image')
        self.make_dir(roi_flow_image_path)

        image_file_list = os.listdir(flow_image_path)
        image_file_list = sorted(image_file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 先排序
        image_file_list = [os.path.join(flow_image_path, f) for f in image_file_list]  # 这里读取出来的图片顺序不对

        for i in range(200):
            each_image = cv2.imread(image_file_list[i])
            roi = calibration_roi[str(i)]

            # 遍历字典中的坐标并绘制矩形框
            for key, (x1, y1, x2, y2) in roi.items():
                start_point = (int(x1), int(y1))
                end_point = (int(x2), int(y2))
                color = (0, 0, 255)  # 红色
                thickness = 3
                cv2.rectangle(each_image, start_point, end_point, color, thickness)

            # 保存标注后的图片
            cv2.imwrite(os.path.join(roi_flow_image_path, f'flow_up_{i}.jpg'), each_image)

    @print_function_name_decorator
    def calc_factor(self, flag, model_name, data_type, scene):
        # 组装点
        points = {}
        physical_distance = 110
        if data_type == 'five_floors_frameworks_360_640':
            if scene == 'normal':
                points = {'point1': [118, 243], 'point2': [166, 243]}
            elif scene == 'light':
                points = {'point1': [108, 210], 'point2': [156, 210]}
            elif scene == 'rain':
                points = {'point1': [116, 92], 'point2': [164, 92]}
        elif data_type == 'five_floors_frameworks_540_960':
            if scene == 'normal':
                points = {'point1': [181, 185], 'point2': [256, 185]}
            elif scene == 'light':
                points = {'point1': [165, 130], 'point2': [240, 130]}
            elif scene == 'rain':
                points = {'point1': [175, 130], 'point2': [250, 130]}

        each_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag])
        items = os.listdir(each_path)
        if 'select_image' not in items:
            # 后面的方法是：直接用预测图片序列的第一张，来取区域
            select_image_path = os.path.join(each_path, 'select_image')
            self.make_dir(select_image_path)

        # 画线
        img = cv2.imread(os.path.join(self.original_dataset_dict[data_type][scene],
                                      os.listdir(self.original_dataset_dict[data_type][scene])[0]))
        cv2.line(img, points['point1'], points['point2'], (255, 0, 0), 3)
        scale_factor_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag],
                                         'select_image', 'scale_factor.png')
        cv2.imwrite(scale_factor_path, img)
        print(f'scale_factor.png保存成功: {scale_factor_path}')

        # 输入已知的物理距离
        pixel_distance = ((points['point1'][0] - points['point2'][0]) ** 2 + (
                points['point1'][1] - points['point2'][1]) ** 2) ** 0.5
        # 计算比例因子
        scale_factor = physical_distance / pixel_distance
        print(f'data_type: {data_type}, scene: {scene}, 缩放因子: {scale_factor}')

        return scale_factor

    @print_function_name_decorator
    def compare_excel(self, flag, model_name, data_type, scene, scale_factor):
        # 加载excel数据
        excel, optical_flow_time = self.get_excel_data(scene)
        # 加载光流数据
        flow_data = self.get_flow_data(flag, model_name, data_type, scene, scale_factor)
        # 画位移图
        self.visualize_displacement(flag, model_name, data_type, scene, flow_data, excel, optical_flow_time)
        # 计算RMSE, NRMSE
        self.calc_error(flag, model_name, data_type, scene, flow_data, excel)
        # 画误差图
        self.visualize_error(flag, model_name, data_type, scene)

    @print_function_name_decorator
    def get_excel_data(self, scene):

        excel_path = None
        optical_flow_time = None
        excel = {}
        if scene == 'normal':
            excel_path = r'original_dataset\excel_data\normal.xlsx'
        elif scene == 'light':
            excel_path = r'original_dataset\excel_data\light.xlsx'
        elif scene == 'fog':
            excel_path = r'original_dataset\excel_data\fog.xlsx'
        elif scene == 'rain':
            excel_path = r'original_dataset\excel_data\rain.xlsx'

        df = pd.read_excel(excel_path)
        if scene == 'normal':
            # 第一个数据是从下标5开始
            excel['time'] = df.iloc[790:1790, 0]
            # optical_flow_time = np.linspace(15, 25, num=200, endpoint=True)
            optical_flow_time = np.linspace(7.9, 17.9, num=200, endpoint=True)
            excel['5'] = df.iloc[790:1790, 11]
            excel['4'] = df.iloc[790:1790, 10]
            excel['3'] = df.iloc[790:1790, 9]
            excel['2'] = df.iloc[790:1790, 8]
            excel['1'] = df.iloc[790:1790, 7]
        elif scene == 'light':
            excel['time'] = df.iloc[530:1530, 0]
            optical_flow_time = np.linspace(5.35, 15.35, num=200, endpoint=True)
            excel['5'] = df.iloc[530:1530, 11]
            excel['4'] = df.iloc[530:1530, 10]
            excel['3'] = df.iloc[530:1530, 9]
            excel['2'] = df.iloc[530:1530, 8]
            excel['1'] = df.iloc[530:1530, 7]
        elif scene == 'rain':
            excel['time'] = df.iloc[900:1900, 0]
            optical_flow_time = np.linspace(9, 19, num=200, endpoint=True)
            excel['5'] = df.iloc[900:1900, 11]
            excel['4'] = df.iloc[900:1900, 10]
            excel['3'] = df.iloc[900:1900, 9]
            excel['2'] = df.iloc[900:1900, 8]
            excel['1'] = df.iloc[900:1900, 7]

        return excel, optical_flow_time

    @print_function_name_decorator
    def get_flow_data(self, flag, model_name, data_type, scene, scale_factor):
        # 加载对应的json文件，即已经识别好的ROI，先使用yolov8n的结果
        filename = os.path.join(r'calibration_roi\five_floors_frameworks_540_960', f'{scene}_yolov8_m_calibration.json')
        with open(filename, 'r', encoding='utf-8') as f:
            calibration_roi = json.load(f)
            f.close()

        # 加载光流数据
        '''
            直接取dx
            flow_data = {'1': [], '2': [], '3': []]}
        '''
        flow_data = {'1': [], '2': [], '3': [], '4': [], '5': []}
        # 读取npz文件
        data_box_list = []
        npz_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'Data_Box')
        npz_items_original = os.listdir(npz_path)
        for npz_item in npz_items_original:
            npz_file_path = os.path.join(npz_path, npz_item)
            data_box_list.append(npz_file_path)
        # 排序
        data_box_list = sorted(data_box_list, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

        for data_box_item in data_box_list:
            npz_data = np.load(data_box_item)
            flow_arr = npz_data['arr']

            for i in range(5):
                for j, each_flow in enumerate(flow_arr):
                    each_flow_roi = calibration_roi[f'{j}'][f'{i + 1}']
                    each_flow_roi_int = [int(coordinate) for coordinate in each_flow_roi]  # 取矩阵的坐标需要是整数
                    mean_displacement = np.mean(flow_arr[j, each_flow_roi_int[1]:each_flow_roi_int[3],
                                                each_flow_roi_int[0]:each_flow_roi_int[2], 0])  # 计算位移平均值
                    flow_data[f'{i + 1}'].append(mean_displacement)

                flow_data[f'{i + 1}'] = np.cumsum(flow_data[f'{i + 1}'])  # 计算累加和，需要加上初始位置的值
                flow_data[f'{i + 1}'] = scale_factor * np.array(flow_data[f'{i + 1}'])  # 乘缩放系数

                # 数据特别处理，normal需要减去3mm, rain需要加上20mm（后面那里要乘-1，方向会反过来）
                if scene == 'normal':
                    flow_data[f'{i + 1}'] = flow_data[f'{i + 1}'] - 2
                elif scene == 'rain':
                    flow_data[f'{i + 1}'] = flow_data[f'{i + 1}'] + 20

                print(len(flow_data[f'{i + 1}']))

        return flow_data

    @print_function_name_decorator
    def visualize_displacement(self, flag, model_name, data_type, scene, flow_data, excel, optical_flow_time):

        plt.figure(figsize=(10, 25))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'  # 设置Times New Roman
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        colors = {'excel': 'black', 'calibration': 'red'}
        linestyles = {'excel': '-', 'calibration': '-.'}
        linewidths = {'excel': 2, 'calibration': 2}
        segments = [str(i + 1) for i in range(5)]
        sub_titles = [f'story{i}' for i in range(len(segments), 0, -1)]

        for i, position in enumerate(segments):

            plt.subplot(len(segments), 1, i + 1)
            plt.plot(excel['time'], excel[position], label='LVDT', linewidth=linewidths['excel'], linestyle='-',
                     color=colors['excel'])

            plt.plot(optical_flow_time, np.array(flow_data[f'{i + 1}']) * (-1),
                     label='calibration', linewidth=linewidths['calibration'],
                     linestyle=linestyles['calibration'], color=colors['calibration'])

            plt.ylabel('Disp[mm]', fontsize=16, fontweight='bold')
            plt.xlabel('times[s]', fontsize=16, fontweight='bold')
            plt.yticks(np.linspace(-50, 50, num=11), fontsize=14)
            plt.legend(loc='upper left', fontsize=14)
            if scene == 'normal':
                plt.xticks(np.linspace(8, 18, num=11), fontsize=14)
            elif scene == 'light':
                plt.xticks(np.linspace(5, 15, num=11), fontsize=14)

            # 设置子图的标题
            plt.title(sub_titles[i], loc='center', fontsize=18, fontweight='bold')

        # 调整子图之间的距离
        plt.tight_layout()
        plt.savefig(os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'displacement.png'))
        plt.show()

    @print_function_name_decorator
    def calc_error(self, flag, model_name, data_type, scene, flow_data, excel):
        '''
            计算excel的每几位的平均值作为一个值
            Industrial_camera: 100fps, 0.01s/frame
            phone: 100fps, 0.01s/frame
            SLR_camera: 1600 => 400
            LVDT: 200fps, 0.005s/frame
        '''
        # 数据采样处理：把LVDT的数据和光流数据的个数变成一样
        excel_processed = {}
        RMSE = {}
        NRMSE = {}
        segments = [str(i + 1) for i in range(5)]

        # 数据降采样
        if scene == 'normal':
            for position in segments:
                Fs = 1000  # 原始采样率
                Fs_new = 200  # 目标采样率
                excel_processed[position] = resample(excel[position], int(len(excel[position]) * Fs_new / Fs))
        elif scene == 'light':
            for position in segments:
                Fs = 1000  # 原始采样率
                Fs_new = 200  # 目标采样率
                excel_processed[position] = resample(excel[position], int(len(excel[position]) * Fs_new / Fs))
        elif scene == 'rain':
            for position in segments:
                Fs = 1000  # 原始采样率
                Fs_new = 200  # 目标采样率
                excel_processed[position] = resample(excel[position], int(len(excel[position]) * Fs_new / Fs))

        # 计算RMSE和NRMSE
        for position in segments:
            RMSE[position] = np.sqrt(np.mean(((excel_processed[position] - flow_data[position] * (-1)) ** 2)))
            NRMSE[position] = RMSE[position] / (np.max(excel_processed[position]) - np.min(excel_processed[position]))

            print(f'scene: {scene}, RMSE: {position, RMSE[position]}, NRMSE: {position, NRMSE[position]}')

        # 保存json文件
        json_RMSE_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'RMSE.json')
        json_NRMSE_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'NRMSE.json')
        with open(json_RMSE_path, "w") as json_file:
            json.dump(RMSE, json_file)
        with open(json_NRMSE_path, "w") as json_file:
            json.dump(NRMSE, json_file)

    @print_function_name_decorator
    def visualize_error(self, flag, model_name, data_type, scene):
        # 读取json文件
        json_RMSE_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'RMSE.json')
        json_NRMSE_path = os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'NRMSE.json')
        with open(json_RMSE_path, "r") as json_file:
            RMSE = json.load(json_file)
        with open(json_NRMSE_path, "r") as json_file:
            NRMSE = json.load(json_file)

        error = {'NRMSE': NRMSE, 'RMSE': RMSE}

        # 创建子图
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'  # 设置英文Times New Roman
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        segments = [str(i + 1) for i in range(5)]
        structures = ['calibration']
        colors = ['red', 'green', 'blue']  # 柱子颜色
        x_ticks = [f'story{i+1}' for i in range(len(RMSE))]

        # 计算柱状图的宽度
        bar_width = 0.3
        x = 0

        for i, evaluation in enumerate(error):

            for j, stage in enumerate(structures):

                # 计算每个柱状图的位置
                x = np.arange(len(segments))
                # 绘制柱状图
                axs[i].bar(x + bar_width * j, list(error[evaluation].values()), width=bar_width, label=stage,
                           color=colors[j])
                if len(structures) == 3:
                    axs[i].set_xticks(x + bar_width)  # 将刻度设置在柱状图的中心
                elif len(structures) == 2:
                    axs[i].set_xticks(x + bar_width / 2)  # 将刻度设置在柱状图的中心
                elif len(structures) == 1:
                    axs[i].set_xticks(x + bar_width / 2)  # 将刻度设置在柱状图的中心
                axs[i].set_xticklabels(x_ticks, fontsize=14)  # 设置刻度标签
                axs[i].tick_params(axis='both', which='major', labelsize=14)

                # 在每个柱子上方添加数值标签
                for k, position in enumerate(error[evaluation]):
                    axs[i].text(x[k] + bar_width * j, error[evaluation][position],
                                f'{error[evaluation][position]:.2f}', ha='center', va='bottom',
                                fontsize=10, fontweight='bold')

            axs[i].set_xlabel('position', fontsize=16, fontweight='bold')
            axs[i].set_ylabel(f'{evaluation}', fontsize=16, fontweight='bold')
            axs[i].legend(loc='upper left', fontsize=14)
            if evaluation == 'RMSE':
                axs[i].set_yticks(np.linspace(0, 20, num=6))
            elif evaluation == 'NRMSE':
                axs[i].set_yticks(np.linspace(0, 0.5, num=11))

        plt.suptitle(f'error', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.original_pretrained_test[model_name][data_type][scene][flag], 'error.png'))
        plt.show()


if __name__ == '__main__':

    # 原来直接用预训练模型来渲染和训练，出来的效果不好
    # 四个数据，两个模型，一个帧不更新
    total_scenes = {'five_floors_frameworks_360_640': ['normal', 'light', 'rain']}
    yolo_model_series = {'v8': ['n', 's', 'm'], 'v9': ['t', 's', 'm']}
    model_list = ['raft-things']
    flag = ['Updated']

    experiment = Experiment(total_scenes, yolo_model_series, model_list, flag)
    experiment.control()


import json
import os
import traceback

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import resample

from function_decorator import print_function_name_decorator


class Experiment:
    
    @print_function_name_decorator
    def __init__(self, total_scenes, yolo_model_series):

        self.total_scenes = total_scenes  # 所有数据集的场景
        self.yolo_model_series = yolo_model_series  # yolo的模型
        self.root = 'yolo_displacement'
        self.path = {}

    @print_function_name_decorator
    def init_experiment(self):

        self.make_dir(self.root)  # 根目录文件夹

        for data_type, scenes in self.total_scenes.items():
            self.path[data_type] = {}
            self.make_dir(os.path.join(self.root, data_type))
            for scene in scenes:
                self.path[data_type][scene] = {}
                self.make_dir(os.path.join(self.root, data_type, scene))
                for series in self.yolo_model_series.keys():
                    self.path[data_type][scene][series] = os.path.join(self.root, data_type, scene, series)
                    self.make_dir(os.path.join(self.root, data_type, scene, series))

    @print_function_name_decorator
    def control(self):
        try:
            self.init_experiment()
            self.result_visualization()

        except Exception as e:
            print("yolo displacement Failure!!!")
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
        for data_type, scenes in self.total_scenes.items():
            for scene in scenes:
                for series in self.yolo_model_series.keys():

                    # 计算缩放因子calc_factor
                    scale_factor = self.calc_factor(data_type, scene, series)

                    for model_type in self.yolo_model_series[series]:
                        # 与excel比较compare_excel
                        self.compare_excel(data_type, scene, series, model_type, scale_factor)

    @print_function_name_decorator
    def calc_factor(self, data_type, scene, series):
        points = {}
        physical_distance = 110
        if data_type == 'five_floors_frameworks_360_640' or data_type == 'five_floors_frameworks_360_640_all':
            if scene == 'normal':
                points = {'point1': [118, 243], 'point2': [168, 243]}
            elif scene == 'light':
                points = {'point1': [108, 210], 'point2': [158, 210]}
            elif scene == 'rain':
                points = {'point1': [116, 92], 'point2': [166, 92]}
        elif data_type == 'five_floors_frameworks_540_960' or data_type == 'five_floors_frameworks_540_960_all':
            if scene == 'normal':
                points = {'point1': [181, 185], 'point2': [256, 185]}
            elif scene == 'light':
                points = {'point1': [165, 130], 'point2': [240, 130]}
            elif scene == 'rain':
                points = {'point1': [175, 130], 'point2': [250, 130]}

        self.make_dir(os.path.join(self.path[data_type][scene][series], 'select_image'))

        # 画线
        image_path = os.path.join('original_dataset', data_type, scene)
        img = cv2.imread(os.path.join(image_path, os.listdir(image_path)[0]))
        cv2.line(img, points['point1'], points['point2'], (255, 0, 0), 3)
        scale_factor_path = os.path.join(self.path[data_type][scene][series], 'select_image', 'scale_factor.png')
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
    def compare_excel(self, data_type, scene, series, model_type, scale_factor):
        # 加载excel数据
        excel, time_data = self.get_excel_data(data_type, scene)
        # 加载光流数据
        yolo_displacement = self.yolo_disp(data_type, scene, series, model_type, scale_factor)
        # 画位移图
        self.visualize_displacement(data_type, scene, series, model_type, yolo_displacement, excel, time_data)
        # 计算RMSE, NRMSE
        self.calc_error(data_type, scene, series, model_type, yolo_displacement, excel)
        # 画误差图
        self.visualize_error(data_type, scene, series, model_type)

    @print_function_name_decorator
    def get_excel_data(self, data_type, scene):
        excel_path = None
        time_data = None
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
            excel['5'] = df.iloc[790:1790, 11]
            excel['4'] = df.iloc[790:1790, 10]
            excel['3'] = df.iloc[790:1790, 9]
            excel['2'] = df.iloc[790:1790, 8]
            excel['1'] = df.iloc[790:1790, 7]
        elif scene == 'light':
            excel['time'] = df.iloc[530:1530, 0]
            excel['5'] = df.iloc[530:1530, 11]
            excel['4'] = df.iloc[530:1530, 10]
            excel['3'] = df.iloc[530:1530, 9]
            excel['2'] = df.iloc[530:1530, 8]
            excel['1'] = df.iloc[530:1530, 7]
        elif scene == 'rain':
            excel['time'] = df.iloc[925:1925, 0]
            excel['5'] = df.iloc[925:1925, 11]
            excel['4'] = df.iloc[925:1925, 10]
            excel['3'] = df.iloc[925:1925, 9]
            excel['2'] = df.iloc[925:1925, 8]
            excel['1'] = df.iloc[925:1925, 7]

        if data_type == 'five_floors_frameworks_360_640' or data_type == 'five_floors_frameworks_540_960':
            if scene == 'normal':
                time_data = np.linspace(7.88, 17.88, num=200, endpoint=True)
            elif scene == 'light':
                time_data = np.linspace(7.88, 17.88, num=200, endpoint=True)
            elif scene == 'rain':
                time_data = np.linspace(9.21, 19.21, num=200, endpoint=True)
        elif data_type == 'five_floors_frameworks_360_640_all' or data_type == 'five_floors_frameworks_540_960_all':
            if scene == 'normal':
                time_data = np.linspace(7.88, 17.88, num=1000, endpoint=True)
            elif scene == 'light':
                time_data = np.linspace(5.3, 15.3, num=1000, endpoint=True)
            elif scene == 'rain':
                time_data = np.linspace(9.21, 19.21, num=1000, endpoint=True)

        return excel, time_data

    @print_function_name_decorator
    def yolo_disp(self, data_type, scene, series, model_type, scale_factor):

        yolo_displacement = {'1': [], '2': [], '3': [], '4': [], '5': []}
        storys = [str(i+1) for i in range(5)]

        # 加载对应的json文件，即已经识别好的ROI，先使用yolov8n的结果
        filename = os.path.join(r'calibration_roi', data_type, f'{scene}_yolo{series}_{model_type}_calibration.json')
        with open(filename, 'r', encoding='utf-8') as f:
            calibration_roi = json.load(f)
            f.close()

        # 计算每一时间点，每一楼层的位移
        for i in range(len(calibration_roi)-1):
            bounding_box_01 = calibration_roi[str(i)]
            bounding_box_02 = calibration_roi[str(i+1)]
            for story in storys:
                points_01 = bounding_box_01[story]
                points_02 = bounding_box_02[story]

                # 计算总位移，包括水平和竖直位移
                # pixel_displacement = ((points_01[0] - points_02['point2'][0]) ** 2
                # + (points_01[1] - points_02[1]) ** 2) ** 0.5
                # 取出前两个坐标x1, y1做差，只算水平位移，用后一个坐标减去前一个坐标，位移计算结果具有正负
                pixel_displacement = points_02[0] - points_01[0]
                yolo_displacement[story].append(pixel_displacement)

        # 位移叠加
        for story in storys:
            yolo_displacement[story] = np.cumsum(yolo_displacement[story])  # 计算累加和，需要加上初始位置的值
            yolo_displacement[story] = scale_factor * np.array(yolo_displacement[story])  # 乘缩放系数

            # 数据特别处理，normal需要减去3mm, rain需要加上20mm（后面那里要乘-1，方向会反过来）
            if scene == 'normal':
                yolo_displacement[story] = yolo_displacement[story] - 2
            elif scene == 'rain':
                yolo_displacement[story] = yolo_displacement[story] + 20
                
        return yolo_displacement

    @print_function_name_decorator
    def visualize_displacement(self, data_type, scene, series, model_type, yolo_displacement, excel, time_data):
        plt.figure(figsize=(10, 25))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'  # 设置Times New Roman
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        colors = {'excel': 'black', 'calibration': 'red'}
        linestyles = {'excel': '-', 'calibration': '-.'}
        linewidths = {'excel': 2, 'calibration': 2}
        storys = [str(i + 1) for i in range(5)]
        sub_titles = [f'story{i}' for i in range(len(storys), 0, -1)]

        for i, story in enumerate(storys):

            plt.subplot(len(storys), 1, i + 1)
            plt.plot(excel['time'], excel[story], label='LVDT', linewidth=linewidths['excel'], linestyle='-',
                     color=colors['excel'])

            plt.plot(time_data, np.array(yolo_displacement[f'{i + 1}']) * (-1),
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
        plt.savefig(os.path.join(self.path[data_type][scene][series], f'{model_type}_displacement.png'))
        plt.show()

    @print_function_name_decorator
    def calc_error(self, data_type, scene, series, model_type, yolo_displacement, excel):

        # 数据采样处理：把LVDT的数据和光流数据的个数变成一样
        excel_processed = {}
        RMSE = {}
        NRMSE = {}
        storys = [str(i + 1) for i in range(5)]
        # 数据降采样
        if data_type == 'five_floors_frameworks_360_640' or data_type == 'five_floors_frameworks_540_960':
            for story in storys:
                Fs = 1000  # 原始采样率
                Fs_new = 200  # 目标采样率
                excel_processed[story] = resample(excel[story], int(len(excel[story]) * Fs_new / Fs))
        elif data_type == 'five_floors_frameworks_360_640_all' or data_type == 'five_floors_frameworks_540_960_all':
            excel_processed = excel

        # 计算RMSE和NRMSE
        for story in storys:
            RMSE[story] = np.sqrt(np.mean((excel_processed[story] - yolo_displacement[story] * (-1)) ** 2))
            NRMSE[story] = RMSE[story] / (np.max(excel_processed[story]) - np.min(excel_processed[story]))
            print(f'scene: {scene}, RMSE: {story, RMSE[story]}, NRMSE: {story, NRMSE[story]}')

        # 保存json文件
        json_RMSE_path = os.path.join(self.path[data_type][scene][series], f'{model_type}_RMSE.json')
        json_NRMSE_path = os.path.join(self.path[data_type][scene][series], f'{model_type}_NRMSE.json')
        with open(json_RMSE_path, "w") as json_file:
            json.dump(RMSE, json_file)
        with open(json_NRMSE_path, "w") as json_file:
            json.dump(NRMSE, json_file)

        # 保存为excel文件
        df_excel_processed = pd.DataFrame(excel_processed)
        df_yolo_displacement = pd.DataFrame(yolo_displacement)

        df_excel_processed.to_excel(os.path.join(self.path[data_type][scene][series], f'{model_type}_excel_processed.xlsx'), index=True)
        df_yolo_displacement.to_excel(os.path.join(self.path[data_type][scene][series], f'{model_type}_yolo_displacement.xlsx'), index=True)

    @print_function_name_decorator
    def visualize_error(self, data_type, scene, series, model_type):
        # 读取json文件
        json_RMSE_path = os.path.join(self.path[data_type][scene][series], f'{model_type}_RMSE.json')
        json_NRMSE_path = os.path.join(self.path[data_type][scene][series], f'{model_type}_NRMSE.json')
        with open(json_RMSE_path, "r") as json_file:
            RMSE = json.load(json_file)
        with open(json_NRMSE_path, "r") as json_file:
            NRMSE = json.load(json_file)

        error = {'NRMSE': NRMSE, 'RMSE': RMSE}

        # 创建子图
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'  # 设置英文Times New Roman
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        storys = [str(i + 1) for i in range(5)]
        structures = ['calibration']
        colors = ['red', 'green', 'blue']  # 柱子颜色
        x_ticks = [f'story{i + 1}' for i in range(len(RMSE))]

        # 计算柱状图的宽度
        bar_width = 0.3
        x = 0

        for i, evaluation in enumerate(error):
            for j, stage in enumerate(structures):

                # 计算每个柱状图的位置
                x = np.arange(len(storys))
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
        plt.savefig(os.path.join(self.path[data_type][scene][series], f'{model_type}_error.png'))
        plt.show()


if __name__ == '__main__':

    # 原来直接用预训练模型来渲染和训练，出来的效果不好
    # 四个数据，两个模型，一个帧不更新
    # total_scenes = {'five_floors_frameworks_360_640_all': ['normal', 'light', 'rain'],
    #                 'five_floors_frameworks_540_960_all': ['normal', 'light', 'rain']}

    total_scenes = {'five_floors_frameworks_540_960_all': ['normal', 'light', 'rain']}

    yolo_model_series = {'v8': ['n', 's', 'm'], 'v9': ['t', 's', 'm']}

    experiment = Experiment(total_scenes, yolo_model_series)
    experiment.control()
    
import argparse
import os

import numpy as np
import cv2
import torch
from tqdm import tqdm

from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder
from RAFT.core.raft import RAFT

import my_dataset
import traceback
from function_decorator import print_function_name_decorator


class Experiment:

    @print_function_name_decorator
    def __init__(self, total_scenes, model_list, flag):

        self.total_scenes = total_scenes  # 所有数据集的场景
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
    def control(self):
        try:
            self.init_experiment()
            self.optical_flow_test()

        except Exception as e:
            print("RAFT Failure!!!")
            print(e)
            traceback.print_exc()

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
    def optical_flow_test(self):
        # 迭代循环的生成数据集，训练模型，以及模型重新在原本的任务上做直接预测
        for flag in self.flag:
            for model_name in self.model_list:
                for data_type, scenes in self.total_scenes.items():
                    for scene in scenes:
                        # 一个模型只需要raft-things预测一次，另外单独存放文件夹
                        self.pretrained_test(flag, model_name, data_type, scene)

    @print_function_name_decorator
    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)  # 如果文件路径不存在，则创建它
            print(f"文件路径 {path} 不存在，已成功创建")
        else:
            print(f"文件路径 {path} 已存在")

    @print_function_name_decorator
    def pre_trained_test_config(self, checkpoint_path, save_path):
        parser = argparse.ArgumentParser()
        # RAFT parameters
        parser.add_argument('--model', help="restore checkpoint", default=checkpoint_path)
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
        parser.add_argument('--save_location', help="save the results in local or oss", default='local')
        parser.add_argument('--save_path', help=" local path to save the result", default=save_path)
        parser.add_argument('--iters', help=" kitti 24, sintel 32", default=32)
        parser.add_argument('--gpus', type=int, nargs='+', default=[0])

        args = parser.parse_args()

        return args

    @print_function_name_decorator
    def pretrained_test(self, flag, model_name, data_type, scene):

        checkpoint_path = f'../models/{model_name}.pth'
        print(f'checkpoint_path: {checkpoint_path}')

        # 保存的路径为新建的文件夹，保存图像文件和data_box
        save_path = self.original_pretrained_test[model_name][data_type][scene][flag]
        # 配置预训练模型测试的变量
        self.config['original_pre_trained_test'] = self.pre_trained_test_config(checkpoint_path, save_path)

        # load RAFT model
        model = torch.nn.DataParallel(RAFT(self.config['original_pre_trained_test']))
        # 加载模型, strict=False表示可以不按照完全相同的模型架构来加载，一般用在迁移学习，但是我们没有对模型架构进行过调整，所以用哪个都行
        model.load_state_dict(torch.load(self.config['original_pre_trained_test'].model), strict=False)
        model.cuda()
        model.eval()

        # 源图片
        img_file_path = self.original_dataset_dict[data_type][scene]
        # 存放光流数据的文件
        Data_Box = []

        # 分别创建存放图片和光流数据的文件
        Data_Box_path = os.path.join(save_path, 'Data_Box')
        image_path = os.path.join(save_path, 'image')
        self.make_dir(Data_Box_path)
        self.make_dir(image_path)

        # 使用dataset加载后优化代码
        dataset_structure = None
        if flag == 'notUpdated':
            dataset_structure = my_dataset.StructureNotUpdated(img_file_path)
        elif flag == 'Updated':
            dataset_structure = my_dataset.StructureUpdated(img_file_path)
        with torch.no_grad():
            for val_id in tqdm(range(len(dataset_structure) - 1)):
                img1, img2, _, _ = dataset_structure[val_id]
                img1 = img1[None].cuda()
                img2 = img2[None].cuda()

                padder = InputPadder(img1.shape)
                img1, img2 = padder.pad(img1, img2)

                flow_low, flow_up = model(img1, img2, iters=self.config['original_pre_trained_test'].iters,
                                          test_mode=True)
                flow_data = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flo_img = flow_viz.flow_to_image(flow_data)
                cv2.imwrite(os.path.join(image_path, f'flow_up_{val_id}.png'), flo_img)
                Data_Box.append(flow_data)
                # print(f'Data_box_size: {len(Data_Box)}, type: {type(Data_Box)}')

                if (val_id + 1) % 200 == 0 and val_id > 0:
                    try:
                        # 保存文件名比finetune的多了一个original
                        npz_path = os.path.join(Data_Box_path, f'flow_up_original_data_chunk_{val_id // 200 + 1}.npz')
                        np.savez(npz_path, arr=Data_Box)
                        # 每次存完之后就清空数据
                        Data_Box.clear()
                        print("Section saved array using Numpy's savez method successfully.")
                        print("Data_Box cleared successfully.")
                    except Exception as e:
                        print("An error occurred:", e)

                # 释放GPU内存
                torch.cuda.empty_cache()


if __name__ == '__main__':

    # 原来直接用预训练模型来渲染和训练，出来的效果不好
    # 四个数据，一个模型，两个策略：帧不更新与帧更新
    total_scenes = {'five_floors_frameworks_540_960': ['normal', 'light', 'rain']}
    model_list = ['raft-things']
    flag = ['notUpdated', 'Updated']

    experiment = Experiment(total_scenes, model_list, flag)
    experiment.control()

# 位移测量网页开发

## Introduction

基于YOLO，Optical Flow算法（RAFT）的位移测量平台。

核心算法：**[ultralytics](https://github.com/ultralytics/ultralytics)**，**[RAFT](https://github.com/princeton-vl/RAFT)**

开发技术：

- 后端：Python Flask
- 前端：html & css & JavaScript

## TODO

Python flask做后端，使用Vue做前端页面设计，基于yolo做结构位移测量的单页应用，类似的应用：https://github.com/Sharpiless/Yolov5-Flask-VUE

- [x] 模型训练和准备，后端算法
  - [x] YOLOv8-v10，前三个大小的模型：`ultralytics\weights`
  - [x] 光流模型：`RAFT\weights`
  - [ ] 后端算法： [core_algorithm](core_algorithm) 
    - [ ] YOLO测试算法：前后端接口已编写完成
    - [ ] 光流测试算法：算法测试demo，未编写完接口
    - [ ] yolo_displacement.py：使用yolo直接计算位移，未完成接口编写
    - [ ] result_visualization_updated：将yolo算法计算出的ROI，与光流图可视化，并计算位移，未完成接口编写
- [ ] 位移结果计算可视化
- [ ] ...

## Contributing or learning

- 本项目使用技术简单，作为网页开发，深度学习入门友好。
- 项目作者未来发展方向不在此方向，目前也没有时间按和经历继续开发，感兴趣的伙伴可以接力，作为学习项目


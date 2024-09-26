from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import os

from core_algorithm.yolo_test import ExperimentYOLO


app = Flask(__name__)
socketio = SocketIO(app)
app.config['UPLOAD_FOLDER'] = './static/uploads'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'message': '没有找到视频文件'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'message': '未选择文件'}), 400

    # 保存上传视频
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    start_time = request.form.get('start_time')
    end_time = request.form.get('end_time')
    video_filename = request.form.get('video_filename')

    # 在这里处理视频的起始和结束时间
    print(f'Start Time: {start_time}, End Time: {end_time}, video_filename: {video_filename}')

    return jsonify({'message': '视频上传成功', 'filename': file.filename}), 200


@app.route('/yolo_detection', methods=['POST'])
def yolo_detection():
    # 获取视频文件名，用户选择的模型
    data = request.get_json()  # 获取请求的 JSON 数据
    video_filename = data.get('filename')
    selected_model = data.get('model')  # 获取选择的模型
    start_time = request.json.get('start_time')
    end_time = request.json.get('end_time')
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

    print("Received data:", data)  # 检查接收到的数据
    print(f"start_time: {start_time}")
    print(f"end_time: {end_time}")
    experiment_yolo = ExperimentYOLO(video_filename, video_path, selected_model, start_time, end_time, socketio)
    experiment_yolo.control()

    annotated_video_filename = f"{video_filename.split('.')[0]}/{selected_model.split('.')[0]}/annotated_video.mp4"  # 生成视频的相对路径
    return jsonify({'message': '计算完成', 'video_filename': annotated_video_filename})


@app.route('/optical_flow', methods=['POST'])
def optical_flow():
    # 实现光流识别算法（这里是一个示例，具体实现需要根据需求）
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], request.json['filename'])

    # 这里调用光流算法处理视频
    # results = optical_flow_algorithm(video_path)  # 实现你的光流算法
    flow_results = {'message': '光流算法处理成功', 'filename': request.json['filename']}

    return jsonify(flow_results), 200


@app.route('/frame_update', methods=['POST'])
def frame_update():
    # 实现帧更新算法（这里是一个示例，具体实现需要根据需求）
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], request.json['filename'])

    # 这里调用帧更新算法处理视频
    # results = frame_update_algorithm(video_path)  # 实现你的帧更新算法
    update_results = {'message': '帧更新处理成功', 'filename': request.json['filename']}

    return jsonify(update_results), 200


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

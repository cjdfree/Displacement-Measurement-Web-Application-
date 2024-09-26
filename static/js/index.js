// 上传视频并加载在页面中，并绑定后端访问接口
let uploadedVideoFilename = ''; // 用于保存上传的视频文件名
function loadAndUploadVideo(event) {
    const fileInput = document.getElementById('videoUpload');
    const selectedFile = fileInput.files[0];
    uploadedVideoFilename = selectedFile.name; // 保存文件名

    // 显示视频
    const video = document.getElementById('uploadedVideo');
    const source = document.getElementById('videoSource');
    source.src = URL.createObjectURL(selectedFile);
    video.style.display = 'block';
    video.load();

    // 获取起始和结束时间
    const startTime = document.getElementById('startTime').value;
    const endTime = document.getElementById('endTime').value;
    // 上传视频到后端
    const formData = new FormData();
    formData.append('video', selectedFile); // 添加视频文件
    formData.append('video_filename', uploadedVideoFilename); // 添加视频的文件名
    formData.append('start_time', startTime); // 添加起始时间
    formData.append('end_time', endTime); // 添加结束时间

    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message); // 处理成功消息
    })
    .catch(error => {
        console.error('上传失败:', error);
    });
}


// YOLO 检测按钮点击事件，并绑定后端访问接口
document.getElementById('yoloButton').addEventListener('click', function() {
    const selectedModel = document.getElementById('modelSelect').value; // 获取选择的模型
    const startTime = document.getElementById('startTime').value; // 获取起始时间
    const endTime = document.getElementById('endTime').value; // 获取结束时间

    fetch('/yolo_detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: uploadedVideoFilename, // 上传视频的文件名
            model: selectedModel, // 发送模型名称
            start_time: startTime, // 发送起始时间
            end_time: endTime // 发送结束时间
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.detection_results); // 处理检测结果
        if (data.video_filename) {
            playAnnotatedVideo(data.video_filename); // 播放生成的视频
        }
    })
    .catch(error => {
        console.error('检测失败:', error);
    });
});

// 播放生成的视频
function playAnnotatedVideo(videoFilename) {
    const videoElement = document.getElementById('annotatedVideo');
    videoElement.src = `D:/pythonProject/PythonWeb/displacement_measurement/ultralytics/results/${videoFilename}`; // 使用绝对路径
    videoElement.load();  // 确保加载视频
    videoElement.play().catch(error => {
        console.error('播放失败:', error);  // 捕获播放错误
    });
}

// yolo计算过程中绑定进度条
const socket = io();
socket.on('progress', (data) => {
    const progressBar = document.getElementById('progressBar');
    const percentage = (data.current / data.total) * 100;
    progressBar.style.width = percentage + '%';
    progressBar.innerText = Math.round(percentage) + '% 完成';
});


// 光流检测按钮点击事件
document.getElementById('opticalFlowButton').addEventListener('click', function() {
    const startTime = document.getElementById('startTime').value; // 获取起始时间
    const endTime = document.getElementById('endTime').value; // 获取结束时间

    fetch('/optical_flow_detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: uploadedVideoFilename,
            start_time: startTime, // 发送起始时间
            end_time: endTime // 发送结束时间
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.optical_flow_results); // 处理光流检测结果
    })
    .catch(error => {
        console.error('光流检测失败:', error);
    });
});

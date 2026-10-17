# YOLO 装甲板推理作业

## 1. 项目功能
使用 YOLO26n 装甲板模型对视频进行推理，绘制检测框、类别、置信度以及 4 个关键点，并将结果保存为视频。

## 2. 运行环境
- Python 3.14
- PyTorch 2.14.1 (CPU 版本)
- Ultralytics 8.4.177
- OpenCV

## 3. 模型权重准备
请将 `best.pt` 模型文件放置在 `./weights/` 目录下。

## 4. 依赖安装与运行命令
1. 激活虚拟环境：`source venv/bin/activate`
2. 安装依赖：`pip install ultralytics opencv-python -i https://mirrors.aliyun.com/pypi/simple/`
3. 运行推理：`python scripts/infer.py`

## 5. 视频拍摄说明
视频由本人在 D414 实验室使用手机自行拍摄，时长约 10-30 秒，画面包含清晰可见的装甲板。

## 6. 结果展示
推理结果保存在 `./results/output.mp4`。结果视频中包含了检测框、类别、置信度和 4 个关键点。

## 7. 遇到的问题与处理
- 那可太多了，比如先下成了GPU版本，还没下成功。以及ai含量百分百，这样真的没问题吗
- 使用 VMware 共享文件夹在 Windows 和 Linux 之间传输模型和视频文件。PR for armor inference

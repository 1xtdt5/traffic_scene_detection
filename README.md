# 基于 YOLO 的交通场景目标检测系统

基于 Django + Ultralytics 的交通场景目标检测与**多模型对比** Web 系统，支持图片检测、视频检测、摄像头实时检测，并提供历史记录管理与统计报表导出。

## 功能特性

- **四模型对比**：YOLOv8s / YOLOv11s / YOLOv8n / YOLOv11n，支持单模型、双模型及四模型同图对比
- **图片检测**：上传图片输出带框结果，支持置信度阈值调节、拖拽上传
- **视频检测**：上传视频逐帧推理，输出带框视频 + 帧级目标数量/类别统计图表
- **实时检测**：调用电脑/手机摄像头进行实时目标检测（轻量 Nano 模型低延迟）
- **历史记录**：图片/视频检测记录分别归档，支持按模型、类别、时间筛选
- **统计报表**：按时间范围导出 PDF / Excel 检测统计报表
- **用户系统**：注册登录、普通用户与管理员角色、管理员数据看板
- **11 类交通目标**：Vehicle、Bus、Bicycle、Person、Engine、Truck、Tricycle、Obstacle、Pothole、Traffic Light、Traffic Sign

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.10+、Django 4.2 |
| 深度学习 | PyTorch、Ultralytics（YOLOv8 / YOLOv11） |
| 数据库 | MySQL 8.0 |
| 前端 | HTML5、Bootstrap 5、原生 JavaScript、Chart.js |
| 训练平台 | Kaggle GPU（T4），Notebook 全流程可复现 |

## 项目结构

```
traffic_scene_detection/
├── webapp/                      # Django 项目
│   ├── config/                  # 全局配置（settings/urls）
│   ├── accounts/                # 用户注册、登录、管理看板
│   ├── detector/                # 检测主应用
│   │   ├── ml/manager.py        # 模型加载与推理核心（4 模型统一调度）
│   │   ├── ml/visualize.py      # 检测框绘制
│   │   ├── models.py            # 数据库模型
│   │   ├── views.py             # 图片/视频/摄像头/报表视图
│   │   └── templates/           # 前端页面
│   ├── static/                  # CSS / JS / 第三方库
│   ├── media/                   # 上传文件与结果（sample 内含测试图）
│   └── models/                  # 训练好的模型权重（已随仓库提供）
├── kaggle/                      # Kaggle 训练 Notebook（6 个）
├── scripts/                     # 数据准备与 VOC 转换脚本
└── requirements.txt
```

## 模型权重

仓库 `webapp/models/` 已内置交通数据集训练好的权重，clone 后无需训练即可使用：

| 文件 | 模型 | 大小 | 用途 |
|------|------|------|------|
| `yolov8s_traffic_best.pt` | YOLOv8s | 21.5 MB | 高精度图片检测 |
| `yolov11s_traffic_best.pt` | YOLOv11s | 18.3 MB | 高精度图片检测 |
| `yolov8n_traffic_best.pt` | YOLOv8n | 6.0 MB | 视频/实时检测（速度优先） |
| `yolov11n_traffic_best.pt` | YOLOv11n | 5.2 MB | 视频/实时检测（速度优先） |
| `traffic_classes.json` | 11 类标签表 | - | 类别名映射 |

> 若权重文件缺失，系统会自动降级为 COCO 预训练模式（80 类通用目标，仅作演示）。

## 环境准备

1. **Python 3.10+**
2. **MySQL 8.0**，创建数据库：

```sql
CREATE DATABASE traffic_detection DEFAULT CHARACTER SET utf8mb4;
```

3. 如账号密码不同，修改 `webapp/config/settings.py` 中的数据库配置：

```python
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": "traffic_detection",
        "USER": "root",
        "PASSWORD": "123456",   # ← 改成你的 MySQL 密码
        "HOST": "127.0.0.1",
        "PORT": "3306",
    }
}
```

## 安装与运行

```bash
# 1. 克隆项目
git clone https://github.com/1xtdt5/traffic_scene_detection.git
cd traffic_scene_detection

# 2. 创建并激活虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 数据库迁移
cd webapp
python manage.py migrate

# 5.（可选）创建管理员账号
python manage.py createsuperuser

# 6. 启动服务
python manage.py runserver 0.0.0.0:8000
```

浏览器访问 **http://127.0.0.1:8000/** 即可使用。

## 训练流程（Kaggle）

`kaggle/` 目录提供完整可复现的训练 Notebook，数据集使用 Kaggle 公开数据集 `tailength/traffic-object`：

| Notebook | 作用 | 挂载依赖 |
|----------|------|---------|
| `1-data-prepare.ipynb` | 数据清洗、8:2 划分、YOLO/VOC 格式转换、打包 | 挂载数据集 |
| `2-train-yolov8.ipynb` | YOLOv8s 训练 + 超参消融 | 挂载 01 输出 |
| `3-train-yolov11s-train.ipynb` | YOLOv11s 训练 + 超参消融 | 挂载 01 输出 |
| `4-compare-model.ipynb` | 双模型 mAP/P/R/FPS/参数量对比评估 | 挂载 01/02/03 |
| `05-yolov8nt-train.ipynb` | YOLOv8n 轻量模型训练 | 挂载 01 输出 |
| `06-yolo11n-train.ipynb` | YOLOv11n 轻量模型训练 | 挂载 01 输出 |

训练设置：GPU T4 x2、Internet On，通过 Add Input 挂载上游 Notebook 输出。训练产出的 `*_traffic_best.pt` 放入 `webapp/models/` 即自动生效。

## 使用说明

1. 注册账号并登录（或使用管理员账号）
2. 首页选择模型（YOLOv8s / YOLOv11s / 双模型 / 四模型），上传交通场景图片
3. 调节置信度阈值，提交检测并查看双模型对比结果
4. 「视频检测」上传 mp4 视频，等待逐帧处理后查看结果视频与统计图
5. 「实时检测」允许摄像头权限，进行实时检测（推荐选用 Nano 模型）
6. 「历史记录」查看与筛选过往检测，「统计报表」按日期导出 PDF/Excel
7. 管理员账号可访问数据看板查看全站统计

## 系统截图（主要页面）

- 首页：Hero 区 + 模型选择卡片 + 拖拽上传
- 结果页：多模型结果并排，置信度进度条可视化
- 视频结果页：结果视频回放 + 帧级检测数量曲线 + 类别分布
- 实时检测页：摄像头画面叠加检测框
- 报表页：按日期范围筛选并导出
- 管理看板：检测量、模型使用分布、用户统计

## 常见问题

**Q：首次启动模型加载很慢？**
A：PyTorch + Ultralitycs 首次初始化较慢，等待 10~30 秒属正常，之后推理很快。

**Q：没有 GPU 能跑吗？**
A：可以，系统默认使用 CPU，imgsz=320 已针对 CPU 推理优化；图片检测秒级完成，视频检测较慢属正常。

**Q：摄像头打不开？**
A：需在 `localhost` 或 HTTPS 环境下使用，并允许浏览器摄像头权限；手机访问需使用电脑局域网 IP 且配置 HTTPS。

**Q：数据库连接失败？**
A：确认 MySQL 服务已启动、数据库 `traffic_detection` 已创建、`settings.py` 中账号密码正确。

## 许可证

本项目仅用于课程学习与学术研究，数据集版权归原作者所有。

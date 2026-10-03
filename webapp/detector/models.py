# -*- coding: utf-8 -*-
from django.conf import settings
from django.db import models

# 模型标识 -> 展示名（含历史标识的兼容映射）
MODEL_LABELS = {
    "yolov8": "YOLOv8s",
    "yolov8n": "YOLOv8n",
    "yolov11": "YOLOv11s",
    "yolov11n": "YOLOv11n",
    # 历史数据兼容（原 YOLOv5 标识迁移后显示）
    "yolo": "YOLOv11s",
    "yolov5n": "YOLOv11n",
    "ssd": "YOLOv8s",
}

# 模型组: 页面可选值 -> 实际执行的模型标识列表
MODEL_GROUPS = {
    "yolov8": ["yolov8"],
    "yolov8n": ["yolov8n"],
    "yolov11": ["yolov11"],
    "yolov11n": ["yolov11n"],
    "both": ["yolov8", "yolov11"],            # s 系列双模型对比
    "nano": ["yolov8n", "yolov11n"],          # n 系列轻量双模型（视频/实时推荐）
    "all": ["yolov8", "yolov8n", "yolov11", "yolov11n"],  # 四模型全对比
}

# 旧标识 -> 新标识的别名映射（兼容历史数据/前端旧入参）
ALIASES = {
    "ssd": "yolov8",
    "yolo": "yolov11",
    "yolov5n": "yolov11n",
}


class DetectionRecord(models.Model):
    """一次检测请求记录（对应计划书: 检测历史记录存储与查询）"""

    MODEL_CHOICES = [
        ("yolov8", "YOLOv8s"),
        ("yolov8n", "YOLOv8n"),
        ("yolov11", "YOLOv11s"),
        ("yolov11n", "YOLOv11n"),
        ("both", "双模型对比(s)"),
        ("nano", "双模型对比(n)"),
        ("all", "四模型全对比"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="detectionrecord", verbose_name="检测用户",
        null=True, blank=True, default=None)
    image = models.ImageField("原图", upload_to="uploads/%Y%m%d/")
    original_name = models.CharField("原始文件名", max_length=255)
    conf_threshold = models.FloatField("置信度阈值")
    models_used = models.CharField("使用模型", max_length=8, choices=MODEL_CHOICES, default="both")
    created_at = models.DateTimeField("检测时间", auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "检测记录"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"#{self.pk} {self.original_name} ({self.models_used})"

    @property
    def model_keys(self):
        return MODEL_GROUPS.get(self.models_used, [self.models_used])


class DetectionResult(models.Model):
    """单模型检测结果（对应计划书: 检测类别、置信度、坐标信息表格展示）"""

    record = models.ForeignKey(
        DetectionRecord, related_name="results", on_delete=models.CASCADE, verbose_name="检测记录")
    model_name = models.CharField("模型名", max_length=16)  # yolov8 / yolov11 / yolov8n / yolov11n
    result_image = models.ImageField("结果图", upload_to="results/%Y%m%d/")
    num_objects = models.IntegerField("检测目标数", default=0)
    inference_ms = models.FloatField("推理耗时(ms)", default=0)
    detail = models.JSONField("检测明细", default=list)

    class Meta:
        verbose_name = "检测结果"
        verbose_name_plural = verbose_name

    @property
    def model_label(self):
        return MODEL_LABELS.get(self.model_name, self.model_name)

    def __str__(self):
        return f"{self.record_id}-{self.model_label} {self.num_objects}个目标"


class VideoRecord(models.Model):
    """视频检测记录（对应需求：上传视频/摄像头实时检测）"""

    SOURCE_CHOICES = [
        ("upload", "上传视频"),
        ("camera", "摄像头实时"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="videorecord", verbose_name="检测用户",
        null=True, blank=True, default=None)
    source_type = models.CharField("来源", max_length=10, choices=SOURCE_CHOICES, default="upload")
    video_file = models.FileField("原视频", upload_to="videos/%Y%m%d/", blank=True, null=True)
    result_video = models.FileField("检测后视频", upload_to="videos/results/%Y%m%d/", blank=True, null=True)
    original_name = models.CharField("原始文件名", max_length=255, blank=True, default="")
    conf_threshold = models.FloatField("置信度阈值")
    models_used = models.CharField("使用模型", max_length=8, default="nano")
    fps = models.FloatField("视频帧率", default=0)
    total_frames = models.IntegerField("总帧数", default=0)
    processed_frames = models.IntegerField("已处理帧数", default=0)
    total_objects = models.IntegerField("检测目标总数", default=0)
    avg_objects_per_frame = models.FloatField("平均每帧目标数", default=0)
    duration_sec = models.FloatField("视频时长(秒)", default=0)
    inference_ms_avg = models.FloatField("平均单帧推理耗时(ms)", default=0)
    status = models.CharField("状态", max_length=10, default="processing")  # processing/done/error
    created_at = models.DateTimeField("创建时间", auto_now_add=True, db_index=True)
    finished_at = models.DateTimeField("完成时间", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "视频检测记录"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"视频#{self.pk} {self.original_name} ({self.models_used})"

    @property
    def model_keys(self):
        return MODEL_GROUPS.get(self.models_used, [self.models_used])


class VideoFrameResult(models.Model):
    """视频单帧检测结果（用于帧级统计与回放）"""

    record = models.ForeignKey(
        VideoRecord, related_name="frames", on_delete=models.CASCADE, verbose_name="视频记录")
    frame_index = models.IntegerField("帧序号")
    model_name = models.CharField("模型名", max_length=16)
    num_objects = models.IntegerField("目标数", default=0)
    inference_ms = models.FloatField("推理耗时(ms)", default=0)
    detail = models.JSONField("检测明细", default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["frame_index"]
        verbose_name = "视频帧结果"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=["record", "frame_index"]),
        ]

    def __str__(self):
        return f"视频{self.record_id}-帧{self.frame_index} {self.num_objects}个目标"


class Report(models.Model):
    """统计报表（PDF/Excel 文件存档，对应需求：报表导出并入库）"""

    REPORT_TYPE_CHOICES = [
        ("image", "图片检测统计"),
        ("video", "视频检测统计"),
        ("combined", "综合统计"),
    ]
    FORMAT_CHOICES = [
        ("pdf", "PDF"),
        ("excel", "Excel"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="report", verbose_name="生成用户",
        null=True, blank=True, default=None)
    report_type = models.CharField("报表类型", max_length=10, choices=REPORT_TYPE_CHOICES, default="combined")
    file_format = models.CharField("文件格式", max_length=10, choices=FORMAT_CHOICES, default="pdf")
    file = models.FileField("报表文件", upload_to="reports/%Y%m%d/")
    date_from = models.DateField("起始日期", null=True, blank=True)
    date_to = models.DateField("结束日期", null=True, blank=True)
    model_filter = models.CharField("模型筛选", max_length=20, blank=True, default="")
    summary = models.JSONField("统计摘要", default=dict)
    created_at = models.DateTimeField("生成时间", auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "统计报表"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"报表#{self.pk} {self.get_report_type_display()}.{self.file_format}"

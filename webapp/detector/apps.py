# -*- coding: utf-8 -*-
from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"
    verbose_name = "交通目标检测"

    def ready(self):
        # 训推分离规范: 模型权重全局只加载一次（懒加载单例，首次检测时初始化，
        # 后续请求直接复用，禁止重复加载）。设置环境变量 DJANGO_PRELOAD_MODELS=1
        # 可在服务启动时即完成预热。
        import os

        if os.environ.get("DJANGO_PRELOAD_MODELS") == "1":
            try:
                from .ml.manager import ModelManager

                ModelManager.get()
                print("[detector] 模型预热完成")
            except Exception as e:  # 权重缺失/依赖异常不阻塞启动
                print(f"[detector] 模型预热失败(将在首次检测时重试): {e}")

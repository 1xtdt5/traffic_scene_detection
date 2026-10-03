# -*- coding: utf-8 -*-
"""
基于Django的交通场景双模型目标检测对比系统 - 全局配置
数据库: MySQL 8.x（计划书原定SQLite3，按需求调整为MySQL，其余保持不变）
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # webapp/

SECRET_KEY = "django-insecure-traffic-detection-#7v2m4k9q&x^zq!5w0r8d3n1p6y4t2e"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "detector",  # 检测功能App
    "accounts",  # 用户认证与管理App
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# 数据库: MySQL（utf8mb4 完整支持中文与emoji）
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": "traffic_detection",
        "USER": "root",
        "PASSWORD": "123456",
        "HOST": "127.0.0.1",
        "PORT": "3306",
        "OPTIONS": {"charset": "utf8mb4"},
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "Asia/Shanghai"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# 上传限制: 单张图片最大 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

# 模型权重目录（Kaggle 训练导出的最优权重放到 webapp/models/ 后自动切换为交通检测模式）
MODEL_WEIGHTS_DIR = BASE_DIR / "models"

# 本地推理时优先使用的设备: auto/cpu
ML_DEVICE = "auto"

# 认证相关: 未登录用户重定向到登录页
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login/"

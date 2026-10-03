# -*- coding: utf-8 -*-
from django.urls import path

from . import views

app_name = "detector"

urlpatterns = [
    path("", views.index, name="index"),
    path("detect/", views.detect, name="detect"),
    path("result/<int:pk>/", views.result, name="result"),
    path("history/", views.history, name="history"),
    path("history/<int:pk>/delete/", views.history_delete, name="history_delete"),
    path("api/detect/", views.api_detect, name="api_detect"),
    # 视频流检测
    path("video/", views.video_page, name="video"),
    path("video/detect/", views.video_detect, name="video_detect"),
    path("video/result/<int:pk>/", views.video_result, name="video_result"),
    path("video/list/", views.video_list, name="video_list"),
    # 摄像头实时检测
    path("camera/", views.camera_page, name="camera"),
    path("api/camera-frame/", views.camera_frame, name="camera_frame"),
    # 统计报表
    path("report/", views.report_page, name="report"),
    path("report/export/", views.export_report, name="export_report"),
    path("report/<int:pk>/download/", views.report_download, name="report_download"),
]

# -*- coding: utf-8 -*-
"""检测系统视图: 上传检测 / 结果对比 / 历史记录 / JSON API"""
import json
import os
import time

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .ml.manager import MODEL_SPECS, ModelManager
from .ml.visualize import draw_detections
from .models import (ALIASES, MODEL_GROUPS, DetectionRecord, DetectionResult,
                     Report, VideoFrameResult, VideoRecord)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_SIZE = getattr(settings, "DATA_UPLOAD_MAX_MEMORY_SIZE", 10 * 1024 * 1024)
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 视频最大 200MB

# 四模型: yolov8/yolov8n + yolov11/yolov11n (全部 ultralytics 框架)
# 旧标识 'ssd'/'yolo'/'yolov5n' 经 ALIASES 映射到新标识


def _index_context():
    manager = ModelManager.get()
    return {
        "available_models": manager.available_models,
        "model_modes": manager.modes,
        "model_labels": {k: v["label"] for k, v in MODEL_SPECS.items()},
    }


@login_required
def index(request):
    """上传首页: 图片上传+预览、模型选择、置信度阈值调节"""
    return render(request, "detector/index.html", _index_context())


def _validate_image(file_obj):
    """校验上传文件, 返回错误信息或None"""
    import pathlib

    ext = pathlib.Path(file_obj.name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return f"不支持的图片格式 {ext}，仅支持: jpg/jpeg/png/bmp/webp"
    if file_obj.size > MAX_SIZE:
        return f"图片过大（{file_obj.size / 1048576:.1f}MB），最大 10MB"
    return None


@require_POST
@login_required
def detect(request):
    """执行检测: 保存记录 -> 双模型推理 -> 保存结果 -> 跳转结果页"""
    upload = request.FILES.get("image")
    if not upload:
        return render(request, "detector/index.html", {
            **_index_context(), "error": "请先选择要检测的图片",
        }, status=400)
    err = _validate_image(upload)
    if err:
        return render(request, "detector/index.html", {**_index_context(), "error": err}, status=400)

    try:
        conf = float(request.POST.get("conf", 0.45))
    except (TypeError, ValueError):
        conf = 0.45
    conf = min(max(conf, 0.05), 0.95)

    raw_mode = request.POST.get("models", "both")
    mode = ALIASES.get(raw_mode, raw_mode)
    if mode not in MODEL_GROUPS:
        mode = "both"

    from PIL import Image, UnidentifiedImageError

    try:
        pil_img = Image.open(upload)
        pil_img.verify()  # 校验图片完整性
        upload.seek(0)
        pil_img = Image.open(upload).convert("RGB")
    except UnidentifiedImageError:
        return render(request, "detector/index.html",
                      {**_index_context(), "error": "图片文件损坏或不是有效图片"}, status=400)

    manager = ModelManager.get()
    model_names = MODEL_GROUPS[mode]

    record = DetectionRecord.objects.create(
        user=request.user, image=upload, original_name=upload.name,
        conf_threshold=conf, models_used=mode)

    for m in model_names:
        t0 = time.perf_counter()
        try:
            detections = manager.detect(m, pil_img, conf)
        except RuntimeError as e:
            # 模型不可用（如YOLO权重缺失）: 保留记录并提示
            DetectionResult.objects.create(
                record=record, model_name=m, num_objects=0, inference_ms=0.0,
                detail=[{"error": str(e)}])
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result_img = draw_detections(pil_img, detections)
        import io

        buf = io.BytesIO()
        result_img.save(buf, format="JPEG", quality=92)
        from django.core.files.base import ContentFile

        result = DetectionResult(
            record=record, model_name=m, num_objects=len(detections),
            inference_ms=round(elapsed_ms, 1), detail=detections)
        result.result_image.save(f"{record.pk}_{m}.jpg", ContentFile(buf.getvalue()), save=True)

    return redirect("detector:result", pk=record.pk)


@login_required
def result(request, pk):
    """结果展示页: 原图与双模型结果并排对比 + 明细表格"""
    record = get_object_or_404(DetectionRecord, pk=pk)
    results = list(record.results.all())
    # 计算各模型类别统计
    for r in results:
        counts = {}
        for d in r.detail:
            if "class" in d:
                counts[d["class"]] = counts.get(d["class"], 0) + 1
        r.class_counts = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return render(request, "detector/result.html", {"record": record, "results": results})


@login_required
def history(request):
    """历史记录页: 分页 + 按文件名/模型筛选"""
    qs = DetectionRecord.objects.prefetch_related("results").all()
    q = request.GET.get("q", "").strip()
    model = ALIASES.get(request.GET.get("model", "").strip(),
                        request.GET.get("model", "").strip())
    if q:
        qs = qs.filter(original_name__icontains=q)
    if model in MODEL_GROUPS:
        qs = qs.filter(Q(models_used=model) | (Q(models_used__in=("both", "nano", "all")) if model in MODEL_SPECS else Q()))
    paginator = Paginator(qs, 9)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "detector/history.html", {
        "page": page, "q": q, "model": model,
        "querystring": f"q={q}&model={model}",
    })


@require_POST
@login_required
def history_delete(request, pk):
    record = get_object_or_404(DetectionRecord, pk=pk)
    record.image.delete(save=False)
    for r in record.results.all():
        r.result_image.delete(save=False)
    record.delete()
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"ok": True})
    return redirect("detector:history")


@csrf_exempt
@require_POST
def api_detect(request):
    """JSON API: POST image(multipart) + models(yolov8|yolo|both) + conf -> 检测结果JSON
    供测试与二次开发使用，不落库"""
    upload = request.FILES.get("image")
    if not upload:
        return JsonResponse({"code": 400, "msg": "缺少 image 文件"}, status=400)
    err = _validate_image(upload)
    if err:
        return JsonResponse({"code": 400, "msg": err}, status=400)
    try:
        conf = float(request.POST.get("conf", 0.45))
    except (TypeError, ValueError):
        conf = 0.45
    conf = min(max(conf, 0.05), 0.95)
    raw_mode = request.POST.get("models", "both")
    mode = ALIASES.get(raw_mode, raw_mode)
    if mode not in MODEL_GROUPS:
        mode = "both"
    from PIL import Image, UnidentifiedImageError

    try:
        pil_img = Image.open(upload).convert("RGB")
    except UnidentifiedImageError:
        return JsonResponse({"code": 400, "msg": "无效图片"}, status=400)
    manager = ModelManager.get()
    out = {"code": 0, "conf_threshold": conf, "results": {}}
    for m in MODEL_GROUPS[mode]:
        try:
            out["results"][m] = manager.detect(m, pil_img, conf)
        except RuntimeError as e:
            out["results"][m] = {"error": str(e)}
    return JsonResponse(out)


# ==================== 视频流检测 ====================

def _validate_video(file_obj):
    import pathlib

    ext = pathlib.Path(file_obj.name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return f"不支持的视频格式 {ext}，仅支持: mp4/avi/mov/mkv/webm"
    if file_obj.size > MAX_VIDEO_SIZE:
        return f"视频过大（{file_obj.size / 1048576:.1f}MB），最大 200MB"
    return None


def _run_video_detection(record, video_path):
    """同步执行视频逐帧检测：抽帧推理 -> 合成带框视频 -> 统计入库。
    为控制耗时，最多处理 MAX_PROCESS_FRAMES 帧（超出部分等间隔跳帧）。"""
    import cv2
    import numpy as np
    from django.core.files.base import ContentFile
    from django.utils import timezone

    from .ml.manager import ModelManager

    from pathlib import Path
    video_path = Path(video_path)  # 统一为 Path 对象，支持 str/Path 入参

    MAX_PROCESS_FRAMES = 300  # 最多推理 300 帧，保证课程演示响应时间可控
    manager = ModelManager.get()
    model_names = MODEL_GROUPS.get(record.models_used, ["yolov8n", "yolov11n"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        record.status = "error"
        record.save(update_fields=["status"])
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    record.fps = round(fps, 2)
    record.total_frames = total
    record.duration_sec = round(total / fps, 2) if fps else 0

    step = max(1, total // MAX_PROCESS_FRAMES) if total > 0 else 1
    out_path = video_path.parent / f"{record.pk}_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, max(fps / step, 1), (width, height))

    from PIL import Image as PILImage

    total_objects = 0
    infer_times = []
    processed = 0
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 双模型结果叠加绘制（首模型画框，其余模型统计）
        all_dets = []
        for m in model_names:
            t0 = time.perf_counter()
            dets = manager.detect_video_frame(m, pil_img, record.conf_threshold)
            elapsed = (time.perf_counter() - t0) * 1000
            infer_times.append(elapsed)
            VideoFrameResult.objects.create(
                record=record, frame_index=frame_idx, model_name=m,
                num_objects=len(dets), inference_ms=round(elapsed, 1), detail=dets)
            total_objects += len(dets)
            all_dets.extend(dets)
        # 画框（去重：同类别同位置只画一次，简化处理直接画首模型的）
        result_img = draw_detections(pil_img, all_dets[:len(all_dets)])
        writer.write(cv2.cvtColor(np.asarray(result_img), cv2.COLOR_RGB2BGR))
        processed += 1
    cap.release()
    writer.release()

    record.processed_frames = processed
    record.total_objects = total_objects
    record.avg_objects_per_frame = round(total_objects / (processed * len(model_names)), 2) if processed else 0
    record.inference_ms_avg = round(sum(infer_times) / len(infer_times), 1) if infer_times else 0
    record.status = "done"
    record.finished_at = timezone.now()
    if out_path.exists():
        with open(out_path, "rb") as f:
            record.result_video.save(f"{record.pk}_result.mp4", ContentFile(f.read()), save=False)
        out_path.unlink(missing_ok=True)
    record.save()


@require_POST
@login_required
def video_detect(request):
    """上传视频检测：保存视频 -> 逐帧推理 -> 合成结果视频 -> 入库统计"""
    upload = request.FILES.get("video")
    if not upload:
        return render(request, "detector/video.html", {
            **_index_context(), "error": "请先选择要检测的视频"}, status=400)
    err = _validate_video(upload)
    if err:
        return render(request, "detector/video.html", {**_index_context(), "error": err}, status=400)

    try:
        conf = float(request.POST.get("conf", 0.45))
    except (TypeError, ValueError):
        conf = 0.45
    conf = min(max(conf, 0.05), 0.95)
    raw_mode = request.POST.get("models", "nano")
    mode = ALIASES.get(raw_mode, raw_mode)
    if mode not in MODEL_GROUPS:
        mode = "nano"

    record = VideoRecord.objects.create(
        user=request.user, source_type="upload", video_file=upload,
        original_name=upload.name, conf_threshold=conf, models_used=mode)
    try:
        # cv2.VideoCapture 不支持含中文的路径，先复制到系统临时目录（纯 ASCII）再处理
        import shutil
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="yolo_vid_")
        tmp_video = os.path.join(tmp_dir, "input.mp4")
        shutil.copy2(record.video_file.path, tmp_video)
        _run_video_detection(record, tmp_video)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        record.status = "error"
        record.save(update_fields=["status"])
        return render(request, "detector/video.html", {
            **_index_context(), "error": f"视频处理失败: {e}"}, status=500)
    return redirect("detector:video_result", pk=record.pk)


@login_required
def video_page(request):
    """视频检测上传页"""
    return render(request, "detector/video.html", _index_context())


@login_required
def video_result(request, pk):
    """视频检测结果页：播放结果视频 + 帧级统计图表"""
    record = get_object_or_404(VideoRecord, pk=pk)
    frames = list(record.frames.all())
    # 帧级统计曲线数据（Chart.js）
    frame_stats = {}
    for f in frames:
        frame_stats.setdefault(f.model_name, []).append({"x": f.frame_index, "y": f.num_objects})
    # 类别分布
    class_dist = {}
    for f in frames:
        for d in f.detail:
            if "class" in d:
                class_dist[d["class"]] = class_dist.get(d["class"], 0) + 1
    class_dist = dict(sorted(class_dist.items(), key=lambda kv: -kv[1]))
    return render(request, "detector/video_result.html", {
        "record": record,
        "frame_stats_json": json.dumps(frame_stats),
        "class_dist_json": json.dumps(class_dist),
        "model_labels": {k: v["label"] for k, v in MODEL_SPECS.items()},
    })


@login_required
def video_list(request):
    """视频检测历史"""
    qs = VideoRecord.objects.filter(user=request.user) if not request.user.is_staff else VideoRecord.objects.all()
    paginator = Paginator(qs, 9)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "detector/video_list.html", {"page": page})


# ==================== 摄像头实时检测 ====================

@login_required
def camera_page(request):
    """摄像头实时检测页：前端 getUserMedia 采集，Canvas 截帧，AJAX 逐帧推理"""
    return render(request, "detector/camera.html", _index_context())


@require_POST
@login_required
def camera_frame(request):
    """摄像头单帧检测 API：接收截帧图片 -> 推理 -> 返回检测框 JSON（不落库，轻量快速）"""
    upload = request.FILES.get("frame")
    if not upload:
        return JsonResponse({"code": 400, "msg": "缺少 frame"}, status=400)
    try:
        conf = float(request.POST.get("conf", 0.45))
    except (TypeError, ValueError):
        conf = 0.45
    conf = min(max(conf, 0.05), 0.95)
    raw_mode = request.POST.get("models", "nano")
    mode = ALIASES.get(raw_mode, raw_mode)
    if mode not in MODEL_GROUPS:
        mode = "nano"
    from PIL import Image, UnidentifiedImageError

    try:
        pil_img = Image.open(upload).convert("RGB")
    except UnidentifiedImageError:
        return JsonResponse({"code": 400, "msg": "无效帧图像"}, status=400)
    manager = ModelManager.get()
    results = {}
    for m in MODEL_GROUPS[mode]:
        results[m] = manager.detect_video_frame(m, pil_img, conf)
    return JsonResponse({"code": 0, "results": results})


# ==================== 统计报表导出（PDF / Excel，入库存档） ====================

def _collect_stats(date_from, date_to, model_filter, report_type):
    """按筛选条件汇总图片+视频检测统计"""
    img_qs = DetectionRecord.objects.prefetch_related("results").all()
    vid_qs = VideoRecord.objects.prefetch_related("frames").all()
    if date_from:
        img_qs = img_qs.filter(created_at__date__gte=date_from)
        vid_qs = vid_qs.filter(created_at__date__gte=date_from)
    if date_to:
        img_qs = img_qs.filter(created_at__date__lte=date_to)
        vid_qs = vid_qs.filter(created_at__date__lte=date_to)
    if model_filter:
        img_qs = img_qs.filter(models_used=model_filter)
        vid_qs = vid_qs.filter(models_used=model_filter)

    class_dist = {}
    model_stats = {}
    total_objects = 0
    for rec in img_qs:
        for r in rec.results.all():
            label = r.model_label
            st = model_stats.setdefault(label, {"count": 0, "objects": 0, "ms_sum": 0.0})
            st["count"] += 1
            st["objects"] += r.num_objects
            st["ms_sum"] += r.inference_ms
            total_objects += r.num_objects
            for d in r.detail:
                if "class" in d:
                    class_dist[d["class"]] = class_dist.get(d["class"], 0) + 1
    for rec in vid_qs:
        for f in rec.frames.all():
            label = MODEL_SPECS.get(f.model_name, {}).get("label", f.model_name)
            st = model_stats.setdefault(label, {"count": 0, "objects": 0, "ms_sum": 0.0})
            st["count"] += 1
            st["objects"] += f.num_objects
            st["ms_sum"] += f.inference_ms
            total_objects += f.num_objects
            for d in f.detail:
                if "class" in d:
                    class_dist[d["class"]] = class_dist.get(d["class"], 0) + 1
    class_dist = dict(sorted(class_dist.items(), key=lambda kv: -kv[1]))
    return {
        "image_records": img_qs.count(),
        "video_records": vid_qs.count(),
        "total_objects": total_objects,
        "class_dist": class_dist,
        "model_stats": model_stats,
    }


def _render_chart_png(class_dist, model_stats):
    """用 matplotlib 渲染统计图表为 PNG 字节流（嵌入 PDF 报表）"""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # 类别分布柱状图
    if class_dist:
        names = list(class_dist.keys())
        vals = list(class_dist.values())
        axes[0].bar(names, vals, color="#2563eb")
        axes[0].set_title("Class Distribution")
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    else:
        axes[0].text(0.5, 0.5, "No Data", ha="center", va="center")
        axes[0].set_title("Class Distribution")
    # 模型目标数对比
    if model_stats:
        labels = list(model_stats.keys())
        objs = [model_stats[k]["objects"] for k in labels]
        axes[1].bar(labels, objs, color=["#7c3aed", "#a78bfa", "#2563eb", "#60a5fa"][:len(labels)])
        axes[1].set_title("Objects per Model")
        axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    else:
        axes[1].text(0.5, 0.5, "No Data", ha="center", va="center")
        axes[1].set_title("Objects per Model")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _build_pdf(stats, date_from, date_to, model_filter):
    import io

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    # 注册中文字体（Windows 自带微软雅黑/宋体）
    import pathlib
    font_candidates = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                       "C:/Windows/Fonts/simsun.ttc"]
    font_name = "Helvetica"
    for fp in font_candidates:
        if pathlib.Path(fp).exists():
            pdfmetrics.registerFont(TTFont("CNFont", fp))
            font_name = "CNFont"
            break

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    from reportlab.lib.styles import ParagraphStyle
    title_style = ParagraphStyle("t", parent=styles["Title"], fontName=font_name, fontSize=18)
    h_style = ParagraphStyle("h", parent=styles["Heading2"], fontName=font_name, fontSize=13)
    n_style = ParagraphStyle("n", parent=styles["Normal"], fontName=font_name, fontSize=10)

    story = [Paragraph("交通场景目标检测系统 · 统计报表", title_style), Spacer(1, 0.4 * cm)]
    story.append(Paragraph(
        f"统计区间: {date_from or '全部'} ~ {date_to or '全部'}　|　模型筛选: {model_filter or '全部'}　|　"
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", n_style))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("一、总体统计", h_style))
    t = Table([
        ["指标", "数值"],
        ["图片检测记录数", stats["image_records"]],
        ["视频检测记录数", stats["video_records"]],
        ["检测目标总数", stats["total_objects"]],
        ["覆盖类别数", len(stats["class_dist"])],
    ], colWidths=[6 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("二、模型对比统计", h_style))
    rows = [["模型", "检测次数", "目标总数", "平均耗时(ms)"]]
    for label, st in stats["model_stats"].items():
        avg_ms = round(st["ms_sum"] / st["count"], 1) if st["count"] else 0
        rows.append([label, st["count"], st["objects"], avg_ms])
    t2 = Table(rows, colWidths=[4 * cm, 3 * cm, 3 * cm, 3.5 * cm])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("三、可视化图表", h_style))
    chart_png = _render_chart_png(stats["class_dist"], stats["model_stats"])
    story.append(RLImage(io.BytesIO(chart_png), width=16 * cm, height=5.8 * cm))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def _build_excel(stats, date_from, date_to, model_filter):
    import io

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "统计摘要"
    head_fill = PatternFill("solid", fgColor="2563EB")
    head_font = Font(color="FFFFFF", bold=True)
    ws.append(["指标", "数值"])
    for c in ws[1]:
        c.fill = head_fill
        c.font = head_font
    ws.append(["图片检测记录数", stats["image_records"]])
    ws.append(["视频检测记录数", stats["video_records"]])
    ws.append(["检测目标总数", stats["total_objects"]])
    ws.append(["覆盖类别数", len(stats["class_dist"])])
    ws.append(["统计区间", f"{date_from or '全部'} ~ {date_to or '全部'}"])
    ws.append(["模型筛选", model_filter or "全部"])

    ws2 = wb.create_sheet("模型对比")
    ws2.append(["模型", "检测次数", "目标总数", "平均耗时(ms)"])
    for c in ws2[1]:
        c.fill = head_fill
        c.font = head_font
    for label, st in stats["model_stats"].items():
        avg_ms = round(st["ms_sum"] / st["count"], 1) if st["count"] else 0
        ws2.append([label, st["count"], st["objects"], avg_ms])

    ws3 = wb.create_sheet("类别分布")
    ws3.append(["类别", "目标数"])
    for c in ws3[1]:
        c.fill = head_fill
        c.font = head_font
    for cls, cnt in stats["class_dist"].items():
        ws3.append([cls, cnt])

    for sheet in (ws, ws2, ws3):
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width = 18

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


@login_required
def report_page(request):
    """报表导出页：筛选条件表单 + 历史报表明细"""
    reports = Report.objects.filter(user=request.user) if not request.user.is_staff else Report.objects.all()
    paginator = Paginator(reports, 10)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "detector/report.html", {
        "page": page,
        "model_labels": {k: v["label"] for k, v in MODEL_SPECS.items()},
    })


@require_POST
@login_required
def export_report(request):
    """生成报表：按筛选条件统计 -> 生成 PDF/Excel -> 入库存档 -> 提供下载"""
    date_from = request.POST.get("date_from") or None
    date_to = request.POST.get("date_to") or None
    model_filter = request.POST.get("model", "").strip()
    file_format = request.POST.get("format", "pdf")
    report_type = request.POST.get("report_type", "combined")

    stats = _collect_stats(date_from, date_to, model_filter, report_type)
    if file_format == "excel":
        content = _build_excel(stats, date_from, date_to, model_filter)
        ext, mime = "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        content = _build_pdf(stats, date_from, date_to, model_filter)
        ext, mime = "pdf", "application/pdf"

    from django.core.files.base import ContentFile

    report = Report(
        user=request.user, report_type=report_type, file_format=file_format,
        date_from=date_from, date_to=date_to, model_filter=model_filter,
        summary={
            "image_records": stats["image_records"],
            "video_records": stats["video_records"],
            "total_objects": stats["total_objects"],
            "class_count": len(stats["class_dist"]),
        })
    fname = f"report_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
    report.file.save(fname, ContentFile(content), save=True)

    # 直接返回文件下载
    from django.http import FileResponse
    return FileResponse(report.file.open("rb"), as_attachment=True, filename=fname, content_type=mime)


@login_required
def report_download(request, pk):
    """下载历史报表"""
    from django.http import FileResponse

    report = get_object_or_404(Report, pk=pk)
    return FileResponse(report.file.open("rb"), as_attachment=True,
                        filename=f"report_{report.pk}.{report.file_format if report.file_format == 'excel' else 'pdf'}")

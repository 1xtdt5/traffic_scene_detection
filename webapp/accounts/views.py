# -*- coding: utf-8 -*-
"""用户认证视图: 登录/注册/登出 + 管理员数据看板"""
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import UserCreationForm
from django.db.models import Avg, Count, Sum
from django.shortcuts import redirect, render
from django.urls import reverse

from detector.models import DetectionRecord, DetectionResult


def login_view(request):
    """登录视图"""
    if request.user.is_authenticated:
        return redirect("detector:index")
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"欢迎回来, {user.username}!")
            nxt = request.GET.get("next") or request.POST.get("next")
            return redirect(nxt) if nxt else redirect("detector:index")
        messages.error(request, "用户名或密码错误")
    return render(request, "accounts/login.html")


def register_view(request):
    """注册视图"""
    if request.user.is_authenticated:
        return redirect("detector:index")
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"注册成功, 欢迎加入, {user.username}!")
            return redirect("detector:index")
        # 表单校验失败, 展示错误
    else:
        form = UserCreationForm()
    return render(request, "accounts/register.html", {"form": form})


def logout_view(request):
    """登出"""
    logout(request)
    messages.info(request, "已安全退出")
    return redirect("accounts:login")


@login_required
@user_passes_test(lambda u: u.is_staff, login_url="/login/")
def dashboard_view(request):
    """管理员数据看板: 检测统计 / 用户列表 / 最近记录"""
    # 检测统计
    total_records = DetectionRecord.objects.count()
    total_results = DetectionResult.objects.count()
    total_objects = DetectionResult.objects.aggregate(
        total=Sum("num_objects"))["total"] or 0
    avg_latency = DetectionResult.objects.aggregate(
        avg=Avg("inference_ms"))["avg"] or 0

    # 按模型统计
    model_stats = DetectionResult.objects.values("model_name").annotate(
        count=Count("id"),
        avg_ms=Avg("inference_ms"),
        total_objects=Sum("num_objects"),
    ).order_by("model_name")

    # 最近 20 条检测记录
    recent_records = DetectionRecord.objects.prefetch_related("results").order_by("-created_at")[:20]

    # 用户列表（仅管理员可见）
    from django.contrib.auth.models import User
    users = User.objects.annotate(
        record_count=Count("detectionrecord"),
    ).order_by("-is_staff", "-date_joined")

    context = {
        "total_records": total_records,
        "total_results": total_results,
        "total_objects": total_objects,
        "avg_latency": round(avg_latency, 1),
        "model_stats": list(model_stats),
        "recent_records": recent_records,
        "users": users,
    }
    return render(request, "accounts/dashboard.html", context)

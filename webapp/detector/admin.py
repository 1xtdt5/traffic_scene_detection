# -*- coding: utf-8 -*-
from django.contrib import admin

from .models import DetectionRecord, DetectionResult


class DetectionResultInline(admin.TabularInline):
    model = DetectionResult
    extra = 0
    readonly_fields = ("model_name", "num_objects", "inference_ms")


@admin.register(DetectionRecord)
class DetectionRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "original_name", "models_used", "conf_threshold", "created_at")
    list_filter = ("models_used", "created_at")
    search_fields = ("original_name",)
    date_hierarchy = "created_at"
    inlines = [DetectionResultInline]


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ("id", "record", "model_name", "num_objects", "inference_ms")
    list_filter = ("model_name",)

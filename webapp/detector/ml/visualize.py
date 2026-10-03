# -*- coding: utf-8 -*-
"""检测结果可视化: 在原图上绘制检测框与标签（与训练端一致的类别体系）"""
import hashlib

from PIL import Image, ImageDraw, ImageFont

_PALETTE = [
    ("#e6194b", "#ffffff"), ("#3cb44b", "#000000"), ("#4363d8", "#ffffff"),
    ("#f58231", "#000000"), ("#911eb4", "#ffffff"), ("#46f0f0", "#000000"),
    ("#f032e6", "#ffffff"), ("#bcf60c", "#000000"), ("#008080", "#ffffff"),
    ("#9a6324", "#ffffff"), ("#800000", "#ffffff"),
]

_font_cache = {}


def _font(size=16):
    key = size
    if key not in _font_cache:
        try:
            # Windows 中文字体优先（类别名为英文，字体仅影响数字/文本渲染质量）
            _font_cache[key] = ImageFont.truetype("arial.ttf", size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _color_for(class_name):
    idx = int(hashlib.md5(class_name.encode("utf-8")).hexdigest(), 16) % len(_PALETTE)
    return _PALETTE[idx]


def draw_detections(pil_image, detections):
    """绘制检测框, 返回 RGB Image（不修改原图）"""
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        line_w = max(2, min(img.size) // 400)
    except Exception:
        line_w = 2
    font = _font(max(14, min(img.size) // 45))
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color, text_color = _color_for(det["class"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        label = f"{det['class']} {det['confidence']:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 3), label, fill=text_color, font=font)
    return img

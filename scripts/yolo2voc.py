# -*- coding: utf-8 -*-
"""
YOLO -> VOC(Pascal XML) 标注格式转换脚本（对应计划书：成员A - YOLO标注转VOC标注脚本开发）

用途: 将 YOLO txt 标注转换为通用 VOC XML 格式（供 VOC 系模型及旧版 SSDLite 实验使用）；YOLOv5s / YOLOv8s 训练直接使用原始 YOLO txt。
输入: prepare_dataset.py 的输出目录（TrafficDetYolo/{train,val}/{images,labels} + classes）
输出: TrafficDetVOC/
    JPEGImages/               所有图片
    Annotations/              每张图对应一个 VOC XML
    ImageSets/Main/{train,val}.txt

用法:
    python yolo2voc.py --yolo-root <TrafficDetYolo目录> --out <输出目录> [--splits train val]
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_classes(yolo_root: Path) -> list:
    """从 data.yaml 或 classes 文件读取类别名"""
    data_yaml = yolo_root / "data.yaml"
    if data_yaml.exists():
        names, in_names = [], False
        for ln in data_yaml.read_text(encoding="utf-8").splitlines():
            if ln.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                s = ln.strip()
                if s.startswith("- "):
                    names.append(s[2:].strip())
                elif s:
                    break
        if names:
            return names
    cands = list(Path(yolo_root).glob("**/classes.txt"))
    if cands:
        return [ln.strip() for ln in cands[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
    raise FileNotFoundError("未找到类别文件(data.yaml 的 names 或 classes.txt)")


def yolo_line_to_voc(cid, cx, cy, w, h, iw, ih):
    """YOLO归一化(cx,cy,w,h) -> VOC绝对坐标(xmin,ymin,xmax,ymax)，并夹紧到图像边界"""
    bw, bh = w * iw, h * ih
    x1, y1 = cx * iw - bw / 2.0, cy * ih - bh / 2.0
    x2, y2 = x1 + bw, y1 + bh
    xmin = max(0, min(iw - 1, int(round(x1))))
    ymin = max(0, min(ih - 1, int(round(y1))))
    xmax = max(xmin + 1, min(iw, int(round(x2))))
    ymax = max(ymin + 1, min(ih, int(round(y2))))
    return xmin, ymin, xmax, ymax


def build_xml(image_stem, img_path, size, objects, class_names):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "JPEGImages"
    ET.SubElement(ann, "filename").text = img_path.name
    ET.SubElement(ET.SubElement(ann, "path"), "dummy").text = ""
    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "Traffic-Object(YOLO2VOC)"
    size_el = ET.SubElement(ann, "size")
    ET.SubElement(size_el, "width").text = str(size[0])
    ET.SubElement(size_el, "height").text = str(size[1])
    ET.SubElement(size_el, "depth").text = str(size[2])
    ET.SubElement(ann, "segmented").text = "0"
    for cid, (xmin, ymin, xmax, ymax) in objects:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = class_names[cid]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(xmin)
        ET.SubElement(bnd, "ymin").text = str(ymin)
        ET.SubElement(bnd, "xmax").text = str(xmax)
        ET.SubElement(bnd, "ymax").text = str(ymax)
    return ann


def main():
    ap = argparse.ArgumentParser(description="YOLO标注转VOC XML")
    ap.add_argument("--yolo-root", required=True, help="prepare_dataset.py 输出目录")
    ap.add_argument("--out", required=True, help="VOC 输出目录")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    yolo_root, out = Path(args.yolo_root), Path(args.out)
    class_names = read_classes(yolo_root)
    print(f"类别({len(class_names)}): {class_names}")

    jpeg_dir = out / "JPEGImages"
    ann_dir = out / "Annotations"
    sets_dir = out / "ImageSets" / "Main"
    for d in (jpeg_dir, ann_dir, sets_dir):
        d.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        img_dir = yolo_root / split / "images"
        lbl_dir = yolo_root / split / "labels"
        if not img_dir.exists():
            print(f"[警告] 缺少 {img_dir}，跳过 {split}")
            continue
        n_img, n_box, n_skip = 0, 0, 0
        lines = []
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                n_skip += 1
                continue
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                iw, ih = im.size
            objects = []
            for ln in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    continue
                cid = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:])
                box = yolo_line_to_voc(cid, cx, cy, w, h, iw, ih)
                objects.append((cid, box))
                n_box += 1
            if not objects:
                n_skip += 1
                continue
            # 统一转存为 jpg，保证 VOC 目录内格式一致
            dst_img = jpeg_dir / (img_path.stem + ".jpg")
            if img_path.suffix.lower() == ".jpg" or img_path.suffix.lower() == ".jpeg":
                from shutil import copy2
                copy2(img_path, dst_img)
            else:
                with Image.open(img_path) as im:
                    im.convert("RGB").save(dst_img, "JPEG", quality=95)
            depth = 3
            xml = build_xml(img_path.stem, dst_img, (iw, ih, depth), objects, class_names)
            ET.ElementTree(xml).write(ann_dir / (img_path.stem + ".xml"), encoding="utf-8", xml_declaration=True)
            lines.append(img_path.stem)
            n_img += 1
        (sets_dir / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[{split}] 图片: {n_img}  框: {n_box}  跳过: {n_skip}")

    print(f"完成! VOC 根目录: {out.resolve()}")


if __name__ == "__main__":
    main()

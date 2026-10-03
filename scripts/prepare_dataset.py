# -*- coding: utf-8 -*-
"""
Traffic-Object 数据集清洗与划分脚本（对应计划书：成员A - 数据集清洗、8:2 固定种子划分训练/验证集）

数据集来源（Kaggle）: https://www.kaggle.com/datasets/tailength/traffic-object
实际下载结构（COCO 格式，非 YOLO）:
    archive (1)/
      classes.txt                              11 个类别名
      ImBalanced-Data/ImBalanced-Data/
        annotations/
          instances_train.json                 COCO 标注（train: 17039 张 / 67k 框）
          instances_val.json                   COCO 标注（val:   3642 张）
          instances_test.json                  COCO 标注（test:  3691 张）
        train/   *.jpg
        val/     *.jpg
        test/    *.jpg

功能:
    1. 读取 classes.txt + COCO instances_{train,val}.json
    2. 清洗: 剔除图片损坏 / 标签缺失 / 框格式非法 / 类别越界 / 坐标越界的样本
    3. COCO bbox [x,y,w,h] (绝对像素) -> YOLO [cx,cy,w,h] (归一化)
    4. 抽样(可选): --sample N 时按固定种子随机抽取 N 张，控制课程实验时长
    5. 划分: 合并 train+val，按 8:2 固定随机种子 (seed=42) 重新划分训练/验证集
       (test 集 <split=auto 时原样复制为 test/，供最终评估使用)
    6. 输出: TrafficDetYolo/{train,val,test}/{images,labels} + data.yaml
            + split_manifest.json（含划分清单与类别分布统计）

用法:
    python prepare_dataset.py --src <数据集根目录> --out <输出目录> [--sample 8000] [--val-ratio 0.2] [--seed 42] [--no-test]

本地示例（典型）:
    python scripts/prepare_dataset.py --src "D:/archive (1)" --out D:/TrafficDetYolo --sample 8000
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_classes(src: Path) -> list:
    """从 classes.txt 读取 11 个类别名（保留出现顺序，作为类别 id）"""
    classes_file = None
    for cand in (src / "classes.txt",
                 src / "ImBalanced-Data" / "ImBalanced-Data" / "classes.txt"):
        if cand.exists():
            classes_file = cand
            break
    if classes_file is None:
        raise FileNotFoundError("未找到 classes.txt，请确认 --src 指向数据集根目录")
    names = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names


def find_coco_root(src: Path) -> tuple:
    """定位 COCO 数据根（含 annotations/ 与 train/ val/ test/）。
    返回 (coco_root, annotations_dir, classes_file_dir)"""
    candidates = [
        src,                                  # 直接是 ImBalanced-Data/ImBalanced-Data/
        src / "ImBalanced-Data" / "ImBalanced-Data",
    ]
    for c in candidates:
        if (c / "annotations").is_dir() and (c / "train").is_dir():
            return c, c / "annotations", src
    raise FileNotFoundError(
        f"未在 {src} 下找到 annotations/ 与 train/ 子目录，请确认 --src 指向解压后的 archive (1) 目录")


def load_coco(ann_path: Path) -> dict:
    """加载单个 COCO instances_*.json, 返回标准化结构"""
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    img_by_id = {im["id"]: im for im in data["images"]}
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(
        sorted(data["categories"], key=lambda c: c["id"]))}
    anns_by_img = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    return {
        "images": img_by_id,
        "cat_id_to_idx": cat_id_to_idx,
        "anns_by_img": dict(anns_by_img),
        "n_cats": len(data["categories"]),
    }


def coco_to_yolo(img_w: int, img_h: int, bbox: list) -> list:
    """COCO [x, y, w, h] (绝对像素, 左上角) -> YOLO [cx, cy, w, h] (归一化, 中心点)"""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return [cx, cy, w / img_w, h / img_h]


def clean_one_coco(img_path: Path, coco: dict, img_id: int, num_classes: int) -> tuple:
    """校验单张图片 + COCO 标注，返回 (是否有效, YOLO 框列表, 原因)"""
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            return False, [], "invalid-size"
    except Exception:
        return False, [], "corrupt-image"

    anns = coco["anns_by_img"].get(img_id, [])
    if not anns:
        return False, [], "empty-label"

    yolo_boxes = []
    for a in anns:
        if a.get("iscrowd", 0):
            continue
        if a["category_id"] not in coco["cat_id_to_idx"]:
            return False, [], "class-out-of-range"
        cid = coco["cat_id_to_idx"][a["category_id"]]
        if not (0 <= cid < num_classes):
            return False, [], "class-out-of-range"
        bx, by, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            continue  # 跳过退化框（不视为整张图损坏）
        cx, cy, nw, nh = coco_to_yolo(w, h, a["bbox"])
        if nw <= 0 or nh <= 0 or nw > 1.0 or nh > 1.0:
            return False, [], "box-out-of-range"
        yolo_boxes.append((cid, cx, cy, nw, nh))

    if not yolo_boxes:
        return False, [], "empty-label"
    return True, yolo_boxes, "ok"


def process_split(coco_root: Path, ann_file: str, images_subdir: str,
                  num_classes: int) -> tuple:
    """处理一个 COCO split（train/val/test），返回 (kept 列表, stats 统计)"""
    ann_path = coco_root / "annotations" / ann_file
    if not ann_path.exists():
        return [], {"missing-ann": 1}
    coco = load_coco(ann_path)
    images_dir = coco_root / images_subdir

    kept, stats = [], defaultdict(int)
    for img_id, img_info in coco["images"].items():
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            stats["missing-image"] += 1
            continue
        ok, boxes, reason = clean_one_coco(img_path, coco, img_id, num_classes)
        if ok:
            kept.append((img_path, img_info["file_name"], boxes))
        else:
            stats[reason] += 1
    return kept, dict(stats)


def copy_with_yolo_label(items: list, out_split_dir: Path, class_names: list) -> dict:
    """复制图片 + 写出 YOLO txt 标签，返回类别分布统计"""
    (out_split_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_split_dir / "labels").mkdir(parents=True, exist_ok=True)
    dist = {c: 0 for c in class_names}
    for img_path, fname, boxes in items:
        shutil.copy2(img_path, out_split_dir / "images" / fname)
        stem = Path(fname).stem
        lines = []
        for cid, cx, cy, w, h in boxes:
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            dist[class_names[cid]] += 1
        (out_split_dir / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    return dist


def main():
    ap = argparse.ArgumentParser(description="Traffic-Object 数据集清洗+划分（COCO 格式）")
    ap.add_argument("--src", required=True,
                    help="数据集根目录（archive (1) 解压后目录，含 classes.txt 与 ImBalanced-Data/）")
    ap.add_argument("--out", required=True, help="输出目录 TrafficDetYolo")
    ap.add_argument("--sample", type=int, default=0,
                    help="抽样数量（仅作用于 train+val 合并集），0 表示使用全部清洗后数据")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="验证集比例，默认 0.2（8:2）")
    ap.add_argument("--seed", type=int, default=42, help="固定随机种子，默认 42")
    ap.add_argument("--no-test", action="store_true",
                    help="不复制 test 集（默认会原样复制为 test/，供最终评估使用）")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    class_names = parse_classes(src)
    num_classes = len(class_names)
    print(f"[1/6] 类别数: {num_classes} -> {class_names}")

    coco_root, ann_dir, _ = find_coco_root(src)
    print(f"[2/6] COCO 数据根: {coco_root}")

    # 处理 train + val
    train_kept, train_stats = process_split(
        coco_root, "instances_train.json", "train", num_classes)
    print(f"    train: 保留 {len(train_kept)} 张，剔除 {train_stats}")
    val_kept, val_stats = process_split(
        coco_root, "instances_val.json", "val", num_classes)
    print(f"    val:   保留 {len(val_kept)} 张，剔除 {val_stats}")

    # 合并后 8:2 划分（按计划书要求，固定种子 seed=42）
    all_kept = train_kept + val_kept
    print(f"[3/6] 合并总数: {len(all_kept)} 张，开始 8:2 划分 (seed={args.seed})")

    rng = random.Random(args.seed)
    if args.sample and args.sample < len(all_kept):
        all_kept = rng.sample(all_kept, args.sample)
        print(f"    抽样: 使用 {args.sample} 张（seed={args.seed}）")

    rng.shuffle(all_kept)
    n_val = max(1, int(len(all_kept) * args.val_ratio))
    val_set, train_set = all_kept[:n_val], all_kept[n_val:]
    print(f"    划分结果: train={len(train_set)}  val={len(val_set)}")

    # 复制 train / val
    print("[4/6] 写出 train / val ...")
    train_dist = copy_with_yolo_label(train_set, out / "train", class_names)
    val_dist = copy_with_yolo_label(val_set, out / "val", class_names)

    # test（原样复制，不做划分）
    test_files = []
    if not args.no_test:
        test_kept, test_stats = process_split(
            coco_root, "instances_test.json", "test", num_classes)
        print(f"    test: 保留 {len(test_kept)} 张，剔除 {test_stats}")
        if test_kept:
            copy_with_yolo_label(test_kept, out / "test", class_names)
            test_files = [f for _, f, _ in test_kept]

    # data.yaml（YOLOv5 训练直接可用）
    print("[5/6] 写出 data.yaml 与 split_manifest.json")
    yaml_text = (
        f"path: {out.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {num_classes}\n"
        "names:\n" + "".join(f"  - {n}\n" for n in class_names)
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "sample": args.sample,
        "total_kept": len(all_kept),
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_files),
        "removed": {**train_stats, **val_stats},
        "class_distribution_train": train_dist,
        "class_distribution_val": val_dist,
        "train_files": [f for _, f, _ in train_set],
        "val_files": [f for _, f, _ in val_set],
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[6/6] 完成! YOLO 数据根目录: {out.resolve()}")
    print(f"      train: {len(train_set)} / val: {len(val_set)} / test: {len(test_files)}")


if __name__ == "__main__":
    main()

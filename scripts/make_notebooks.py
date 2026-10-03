# -*- coding: utf-8 -*-
"""
将 kaggle/nb_src/*.py（# %% 单元格标记格式）转换为可直接上传 Kaggle 的 .ipynb 文件。

约定:
  - "# %%"            : 开始一个 code 单元格
  - "# %% [md]"       : 开始一个 markdown 单元格
  - "# !" 开头的行    : 转换后还原为 "!..."  (shell 命令)
  - "# %" 开头的行    : 转换后还原为 "%..."  (IPython 魔法命令, 如 %pip / %%writefile)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "kaggle" / "nb_src"
OUT_DIR = ROOT / "kaggle"


def is_cell_marker(line: str) -> bool:
    s = line.strip()
    if not s.startswith("# %%"):
        return False
    rest = s[4:]  # 去掉 "# %%"
    return rest == "" or rest.startswith(" ")  # "# %%" 或 "# %% [md]"，排除 "# %%writefile"


def build_notebook(cells):
    nb_cells = []
    for ctype, lines in cells:
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()
        if not lines:
            continue
        if ctype == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]})
        else:
            out = []
            for ln in lines:
                if ln.lstrip().startswith("# !"):
                    ln = ln.replace("# !", "!", 1)
                elif ln.lstrip().startswith("# %"):
                    ln = ln.replace("# %", "%", 1)
                out.append(ln)
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": [ln + "\n" for ln in out[:-1]] + [out[-1]]})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert(py_path: Path):
    lines = py_path.read_text(encoding="utf-8").splitlines()
    cells, cur, ctype = [], None, None
    for line in lines:
        if is_cell_marker(line):
            if cur is not None:
                cells.append((ctype, cur))
            ctype = "markdown" if "[md]" in line else "code"
            cur = []
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        cells.append((ctype, cur))
    nb = build_notebook(cells)
    out_path = OUT_DIR / (py_path.stem + ".ipynb")
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{py_path.name} -> {out_path.name}  ({len(nb['cells'])} cells)")


def main():
    files = sorted(SRC_DIR.glob("*.py"))
    if not files:
        print("no sources"); sys.exit(1)
    for f in files:
        convert(f)


if __name__ == "__main__":
    main()

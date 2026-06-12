"""dataset_repair.py — 数据集自动修复 (随 Orchestrator 启动在 GPU 机上执行)

dataset_inspector 只"报告"问题, 本模块负责"修"。针对 2026-06-10 报告:
  - train/val duplicate filenames: 701  → 训练/验证集泄漏嫌疑
  - invalid bbox rows: 5 / class id out of bounds rows: 8

修复策略 (全部幂等、可 dry-run、绝不删除数据):
  1. train/val 同名文件: 先比大小再比 sha256。内容完全相同 = 真泄漏 →
     把 train 侧副本 (图片+标签) 移入 person_dataset/_quarantine/
     train_val_dup/ (保 val 不动, 保证评估集连续性)。同名但内容不同的
     只报告, 不动 (可能只是命名巧合)。
  2. 标签行修复 (nc=1):
     - class id != 0 → 改写为 0 (与 single_cls=True 训练语义一致, 不丢框)
     - bbox 坐标越界 → clamp 到 [0,1]; clamp 后 w/h<=0 或数值非法 → 删行
     原文件先备份到 _quarantine/label_backup/<split>/。
  3. 有改动时清掉 ultralytics 的 *.cache (它按内容哈希自检, 但显式清理
     更稳), 并写 reports/dataset_repair_report.json 供 dashboard 查看。

运行方式:
  python -m autoresearch_v2.dataset_repair [--data-yaml X] [--dry-run]
  (Orchestrator 启动时自动调用, --skip-dataset-repair 可关)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
))
DEFAULT_DATA_YAML = os.path.join(REPO_ROOT, "person_dataset", "person.yaml")
DEFAULT_REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
QUARANTINE_DIRNAME = "_quarantine"
IMG_EXTS = (".jpg", ".JPG", ".jpeg", ".png", ".PNG", ".bmp")


def _log(msg: str):
    print(f"[dataset_repair] {msg}", flush=True)


def _load_yaml(data_yaml: str) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml not installed")
    with open(data_yaml, "r", encoding="utf-8", errors="replace") as f:
        return yaml.safe_load(f) or {}


def _dataset_root(data_yaml: str, cfg: dict) -> str:
    """与 train_utils.resolve_data_yaml 同源的根目录解析:
    相对 `path:` 按 yaml 自身位置解析, 不依赖 ultralytics datasets_dir。"""
    raw = str(cfg.get("path", "") or "")
    train_rel = str(cfg.get("train", "images/train") or "images/train")
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    candidates = []
    if raw and os.path.isabs(raw):
        candidates.append(raw)
    elif raw:
        candidates.append(os.path.normpath(os.path.join(yaml_dir, raw)))
    candidates.append(yaml_dir)
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, train_rel)):
            return cand
    return yaml_dir


def _split_dirs(root: str, cfg: dict, split: str):
    rel = str(cfg.get(split, f"images/{split}") or f"images/{split}")
    img_dir = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(root, rel))
    label_dir = img_dir.replace(os.sep + "images" + os.sep,
                                os.sep + "labels" + os.sep)
    if label_dir == img_dir:  # 路径里没有 images 段时的兜底
        label_dir = os.path.normpath(os.path.join(root, "labels", split))
    return img_dir, label_dir


def _list_images(img_dir: str) -> dict:
    """stem -> path (同一 stem 多扩展名时取第一个)。"""
    out = {}
    if not os.path.isdir(img_dir):
        return out
    for name in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(name)
        if ext in IMG_EXTS and stem not in out:
            out[stem] = os.path.join(img_dir, name)
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _move_to(path: str, dest_dir: str, dry_run: bool) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(path)
    dest = os.path.join(dest_dir, base)
    n = 1
    while os.path.exists(dest):
        stem, ext = os.path.splitext(base)
        dest = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        n += 1
    if not dry_run:
        shutil.move(path, dest)
    return dest


# ── 1. train/val 泄漏隔离 ─────────────────────────────────────────────

def dedup_train_val(root: str, cfg: dict, dry_run: bool) -> dict:
    train_img_dir, train_label_dir = _split_dirs(root, cfg, "train")
    val_img_dir, _ = _split_dirs(root, cfg, "val")
    train_imgs = _list_images(train_img_dir)
    val_imgs = _list_images(val_img_dir)
    common = sorted(set(train_imgs) & set(val_imgs))

    qdir = os.path.join(root, QUARANTINE_DIRNAME, "train_val_dup")
    quarantined, diff_content = [], []
    for stem in common:
        t_path, v_path = train_imgs[stem], val_imgs[stem]
        try:
            # 先比大小 — 不同则内容必不同, 省掉哈希大文件
            if os.path.getsize(t_path) != os.path.getsize(v_path) or \
                    _sha256(t_path) != _sha256(v_path):
                diff_content.append(stem)
                continue
        except OSError:
            continue
        _move_to(t_path, os.path.join(qdir, "images"), dry_run)
        label_path = os.path.join(train_label_dir, stem + ".txt")
        if os.path.isfile(label_path):
            _move_to(label_path, os.path.join(qdir, "labels"), dry_run)
        quarantined.append(stem)

    return {
        "common_filenames": len(common),
        "identical_quarantined": len(quarantined),
        "quarantined_stems": quarantined[:50],
        "same_name_diff_content": len(diff_content),
        "diff_content_stems": diff_content[:50],
        "quarantine_dir": qdir,
    }


# ── 2. 标签行修复 ─────────────────────────────────────────────────────

def repair_label_file(path: str, nc: int = 1):
    """返回 (fixed_lines, stats) — 不写文件, 纯函数便于单测。"""
    stats = {"class_id_fixed": 0, "bbox_clamped": 0, "rows_dropped": 0}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    fixed, changed = [], False
    for line in raw_lines:
        toks = line.split()
        if not toks:
            changed = changed or line.strip() != ""
            continue
        if len(toks) < 5:
            stats["rows_dropped"] += 1
            changed = True
            continue
        try:
            cls = int(float(toks[0]))
            cx, cy, w, h = (float(t) for t in toks[1:5])
        except ValueError:
            stats["rows_dropped"] += 1
            changed = True
            continue
        if any(v != v or v in (float("inf"), float("-inf"))
               for v in (cx, cy, w, h)):
            stats["rows_dropped"] += 1
            changed = True
            continue

        row_changed = False
        if cls < 0 or cls >= nc:
            if nc == 1:
                cls = 0          # 单类数据集: 错类号是标注笔误, 保框改类
                stats["class_id_fixed"] += 1
                row_changed = True
            else:
                stats["rows_dropped"] += 1
                changed = True
                continue

        clamped = [min(1.0, max(0.0, v)) for v in (cx, cy, w, h)]
        if clamped != [cx, cy, w, h]:
            stats["bbox_clamped"] += 1
            row_changed = True
        cx, cy, w, h = clamped
        if w <= 0 or h <= 0:
            stats["rows_dropped"] += 1
            changed = True
            continue

        rest = toks[5:]
        if row_changed:
            changed = True
            parts = [str(cls)] + [f"{v:.6f}" for v in (cx, cy, w, h)] + rest
            fixed.append(" ".join(parts) + "\n")
        else:
            fixed.append(line if line.endswith("\n") else line + "\n")

    return (fixed if changed else None), stats


def repair_labels(root: str, cfg: dict, dry_run: bool) -> dict:
    nc = int(cfg.get("nc", 1) or 1)
    backup_root = os.path.join(root, QUARANTINE_DIRNAME, "label_backup")
    total = {"files_changed": 0, "class_id_fixed": 0,
             "bbox_clamped": 0, "rows_dropped": 0,
             "backup_dir": backup_root, "unlabeled_splits": {}}

    for split in ("train", "val", "test"):
        img_dir, label_dir = _split_dirs(root, cfg, split)
        if not os.path.isdir(img_dir):
            continue
        label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        n_images = len(_list_images(img_dir))
        if n_images and not label_files:
            # 修不了缺失的标签 — 只能如实报告 (evaluate 已学会跳过)
            total["unlabeled_splits"][split] = n_images
            continue
        for lf in label_files:
            try:
                fixed, stats = repair_label_file(lf, nc=nc)
            except OSError:
                continue
            if fixed is None:
                continue
            total["files_changed"] += 1
            for k in ("class_id_fixed", "bbox_clamped", "rows_dropped"):
                total[k] += stats[k]
            if not dry_run:
                bdir = os.path.join(backup_root, split)
                os.makedirs(bdir, exist_ok=True)
                bpath = os.path.join(bdir, os.path.basename(lf))
                if not os.path.exists(bpath):
                    shutil.copyfile(lf, bpath)
                with open(lf, "w", encoding="utf-8") as f:
                    f.writelines(fixed)
    return total


# ── 3. 缓存清理 ───────────────────────────────────────────────────────

def clear_ultralytics_caches(root: str, cfg: dict, dry_run: bool) -> list:
    """数据被修过后清掉 *.cache — ultralytics 会自动重建。"""
    targets = set(glob.glob(os.path.join(root, "*.cache")))
    for split in ("train", "val", "test"):
        img_dir, label_dir = _split_dirs(root, cfg, split)
        for d in (img_dir, label_dir):
            targets.add(d + ".cache")
            targets.update(glob.glob(os.path.join(os.path.dirname(d), "*.cache")))
    removed = []
    for t in sorted(targets):
        if os.path.isfile(t):
            if not dry_run:
                try:
                    os.remove(t)
                except OSError:
                    continue
            removed.append(t)
    return removed


# ── 入口 ──────────────────────────────────────────────────────────────

def repair_dataset(data_yaml: str = DEFAULT_DATA_YAML,
                   dry_run: bool = False,
                   reports_dir: str = DEFAULT_REPORTS_DIR) -> dict:
    t0 = time.time()
    cfg = _load_yaml(data_yaml)
    root = _dataset_root(data_yaml, cfg)

    dup = dedup_train_val(root, cfg, dry_run)
    labels = repair_labels(root, cfg, dry_run)
    changed = bool(dup["identical_quarantined"] or labels["files_changed"])
    caches = clear_ultralytics_caches(root, cfg, dry_run) if changed else []

    report = {
        "generated_at": datetime.now().isoformat(),
        "data_yaml": data_yaml,
        "dataset_root": root,
        "dry_run": dry_run,
        "duplicates": dup,
        "labels": labels,
        "caches_removed": caches,
        "changed": changed,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    _log(f"root={root} dry_run={dry_run}")
    _log(f"train/val 同名 {dup['common_filenames']} 个: "
         f"内容相同已隔离 {dup['identical_quarantined']}, "
         f"同名不同内容 {dup['same_name_diff_content']} (保留)")
    _log(f"标签修复: {labels['files_changed']} 个文件 "
         f"(类号纠正 {labels['class_id_fixed']} 行, "
         f"坐标clamp {labels['bbox_clamped']} 行, "
         f"删除坏行 {labels['rows_dropped']})")
    if labels["unlabeled_splits"]:
        _log(f"无标签 split (无法自动修, 仅报告): {labels['unlabeled_splits']}")
    if caches:
        _log(f"已清缓存: {caches}")

    try:
        os.makedirs(reports_dir, exist_ok=True)
        out = os.path.join(reports_dir, "dataset_repair_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["report_json"] = out
    except OSError as e:
        _log(f"报告写入失败: {e}")
    return report


def main():
    parser = argparse.ArgumentParser(description="数据集自动修复")
    parser.add_argument("--data-yaml", default=DEFAULT_DATA_YAML)
    parser.add_argument("--dry-run", action="store_true",
                        help="只报告将要做什么, 不动任何文件")
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()
    report = repair_dataset(args.data_yaml, dry_run=args.dry_run,
                            reports_dir=args.reports_dir)
    sys.exit(0 if report else 1)


if __name__ == "__main__":
    main()

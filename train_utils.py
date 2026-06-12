"""train_utils.py — train.py 的轻量辅助 (不 import ultralytics, 可单测)

两个历史翻车点的永久修复:

1. resolve_data_yaml — person.yaml 里 `path:` 是相对路径时, ultralytics
   按它自己 settings.json 的 datasets_dir 解析 (不是按 CWD, 也不是按
   yaml 所在目录), 跨机器随设置漂移 → "Dataset not found" → 训练 rc!=0
   → 实验失败。这里按 yaml 自身位置解析成绝对路径, 写一份解析副本传给
   训练, 与机器/设置彻底解耦。(历史: 7d575c1 / 8e39755 / 35bd028 三次
   修同一类 path 问题。)

2. resolve_model — 旧 fallback 列表漏掉 yolo12x 且按字母序取"最大",
   字母序 ≠ 模型大小 (s > l)。改为显式大小优先级。
"""

from __future__ import annotations

import os

# 显式按模型容量降序 — 不要按字母序 sort (yolo12s 会排在 yolo12l 前面)
MODEL_PREFERENCE = ["yolo12x.pt", "yolo12l.pt", "yolo12m.pt",
                    "yolo12s.pt", "yolo12n.pt"]


def resolve_model(model_name: str, repo_root: str) -> str:
    """模型解析: 指定模型在仓库根目录存在就用它; 不存在则按容量从大到小
    找本地可用的替代, 避免触发 GitHub 在线下载 (断网机上必失败)。"""
    def _abs(name: str) -> str:
        if os.path.isabs(name):
            return os.path.normpath(name)
        return os.path.normpath(os.path.join(repo_root, name))

    model_path = _abs(model_name)
    if os.path.isfile(model_path):
        return model_path

    print(f"[WARN] Model {model_name!r} not found on disk, searching for fallback...")
    for name in MODEL_PREFERENCE:
        if name == model_name:
            continue
        path = _abs(name)
        if os.path.isfile(path):
            print(f"[INFO] Falling back to {name}")
            return path

    raise RuntimeError("No YOLO models found on disk. Cannot train.")


def resolve_data_yaml(data_yaml_path: str, out_dir: str) -> str:
    """把 data yaml 的相对 `path:` 固定为绝对路径, 写解析副本到 out_dir,
    返回副本路径。已是有效绝对路径 / 无法解析时原样返回 (让 ultralytics
    自己尝试, 不比现状更糟)。"""
    try:
        import yaml
    except ImportError:
        return data_yaml_path

    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return data_yaml_path

    raw_path = str(data.get("path", "") or "")
    train_rel = str(data.get("train", "images/train") or "images/train")

    if raw_path and os.path.isabs(raw_path):
        if os.path.isdir(os.path.join(raw_path, train_rel)):
            return data_yaml_path  # 已经是有效绝对路径, 不动
        # 绝对但失效 (机器迁移) — 继续尝试按 yaml 位置重解析

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    candidates = []
    if raw_path and not os.path.isabs(raw_path):
        candidates.append(os.path.normpath(os.path.join(yaml_dir, raw_path)))
    # 数据集通常与 yaml 同目录 (person_dataset/person.yaml + images/)
    candidates.append(yaml_dir)

    resolved = None
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, train_rel)):
            resolved = cand
            break
    if resolved is None:
        return data_yaml_path

    data["path"] = resolved
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "_resolved_data.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    except Exception:
        return data_yaml_path
    print(f"[INFO] data yaml `path:` resolved: {raw_path!r} -> {resolved!r}")
    return out_path

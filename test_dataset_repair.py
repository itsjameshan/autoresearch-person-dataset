"""test_dataset_repair.py — 数据集自动修复的单元测试

对应 2026-06-10 dataset_report 的实际问题:
  - train/val duplicate filenames: 701 (泄漏嫌疑)
  - invalid bbox rows: 5 / class id out of bounds rows: 8
  - test split 431 张图 0 个标签 (修不了, 只能报告 + evaluate 跳过)

运行: python -m pytest test_dataset_repair.py -v
"""

import json
import os

import pytest
import yaml

from autoresearch_v2 import dataset_repair
from autoresearch_v2.dataset_repair import (
    repair_dataset, repair_label_file, dedup_train_val,
)
from autoresearch_v2.orchestrator import Orchestrator


# ════════════════════════════════════════════════════════════════════
# 测试数据集脚手架
# ════════════════════════════════════════════════════════════════════

def make_dataset(tmp_path, *, with_dups=True, with_bad_labels=True,
                 unlabeled_test=True):
    root = tmp_path / "person_dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        if split != "test" or not unlabeled_test:
            (root / "labels" / split).mkdir(parents=True)

    def img(split, name, content: bytes):
        p = root / "images" / split / name
        p.write_bytes(content)
        return p

    def label(split, name, text):
        p = root / "labels" / split / name
        p.write_text(text, encoding="utf-8")
        return p

    # 普通样本
    img("train", "t_only.jpg", b"AAAA")
    label("train", "t_only.txt", "0 0.5 0.5 0.2 0.2\n")
    img("val", "v_only.jpg", b"BBBB")
    label("val", "v_only.txt", "0 0.4 0.4 0.1 0.1\n")

    if with_dups:
        # 内容完全相同的 train/val 同名文件 = 真泄漏
        img("train", "dup_same.jpg", b"SAME_BYTES")
        label("train", "dup_same.txt", "0 0.5 0.5 0.1 0.1\n")
        img("val", "dup_same.jpg", b"SAME_BYTES")
        label("val", "dup_same.txt", "0 0.5 0.5 0.1 0.1\n")
        # 同名但内容不同 = 命名巧合, 必须保留
        img("train", "dup_diff.jpg", b"TRAIN_BYTES")
        img("val", "dup_diff.jpg", b"VAL_OTHER")

    if with_bad_labels:
        label("train", "bad.txt",
              "3 0.5 0.5 0.1 0.1\n"        # 类号越界 (nc=1) → 改 0
              "0 1.2 0.5 0.1 0.1\n"        # cx 越界 → clamp
              "0 0.5 0.5 -0.1 0.1\n"       # w<=0 → 删行
              "garbage line here\n"        # 非法 → 删行
              "0 0.3 0.3 0.05 0.05\n")     # 完好 → 原样保留
        img("train", "bad.jpg", b"BAD_IMG")

    if unlabeled_test:
        img("test", "u1.jpg", b"T1")
        img("test", "u2.jpg", b"T2")

    # ultralytics 缓存
    (root / "labels" / "train.cache").write_bytes(b"cache")

    yaml_path = root / "person.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "path": "person_dataset", "train": "images/train",
            "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "person"},
        }, f)
    return root, str(yaml_path)


# ════════════════════════════════════════════════════════════════════
# train/val 泄漏隔离
# ════════════════════════════════════════════════════════════════════

class TestDedup:
    def test_identical_dup_quarantined_from_train_only(self, tmp_path):
        root, yaml_path = make_dataset(tmp_path)
        report = repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))

        dup = report["duplicates"]
        assert dup["common_filenames"] == 2
        assert dup["identical_quarantined"] == 1
        assert dup["same_name_diff_content"] == 1

        assert not (root / "images" / "train" / "dup_same.jpg").exists()
        assert not (root / "labels" / "train" / "dup_same.txt").exists()
        # val 侧绝不动 (评估集连续性)
        assert (root / "images" / "val" / "dup_same.jpg").exists()
        assert (root / "labels" / "val" / "dup_same.txt").exists()
        # 隔离 = 移动, 不是删除
        q = root / "_quarantine" / "train_val_dup"
        assert (q / "images" / "dup_same.jpg").read_bytes() == b"SAME_BYTES"
        assert (q / "labels" / "dup_same.txt").exists()

    def test_same_name_different_content_kept(self, tmp_path):
        root, yaml_path = make_dataset(tmp_path)
        repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))
        assert (root / "images" / "train" / "dup_diff.jpg").read_bytes() == b"TRAIN_BYTES"
        assert (root / "images" / "val" / "dup_diff.jpg").read_bytes() == b"VAL_OTHER"

    def test_idempotent_second_run(self, tmp_path):
        root, yaml_path = make_dataset(tmp_path)
        repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))
        second = repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))
        assert second["duplicates"]["identical_quarantined"] == 0
        assert second["labels"]["files_changed"] == 0
        assert second["changed"] is False

    def test_dry_run_moves_nothing(self, tmp_path):
        root, yaml_path = make_dataset(tmp_path)
        report = repair_dataset(yaml_path, dry_run=True,
                                reports_dir=str(tmp_path / "reports"))
        assert report["duplicates"]["identical_quarantined"] == 1
        assert (root / "images" / "train" / "dup_same.jpg").exists()
        assert (root / "labels" / "train" / "bad.txt").read_text(
            encoding="utf-8").startswith("3 ")
        assert (root / "labels" / "train.cache").exists()


# ════════════════════════════════════════════════════════════════════
# 标签行修复
# ════════════════════════════════════════════════════════════════════

class TestLabelRepair:
    def test_repair_label_file_rules(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text(
            "3 0.5 0.5 0.1 0.1\n"
            "0 1.2 -0.05 0.1 0.1\n"
            "0 0.5 0.5 0.0 0.1\n"
            "not a number row\n"
            "0 0.3 0.3 0.05 0.05\n",
            encoding="utf-8")
        fixed, stats = repair_label_file(str(p), nc=1)

        assert stats["class_id_fixed"] == 1
        assert stats["bbox_clamped"] == 1
        assert stats["rows_dropped"] == 2
        assert fixed is not None and len(fixed) == 3
        assert fixed[0].startswith("0 0.5")          # 类号 3 → 0
        assert fixed[1].startswith("0 1.000000 0.000000")  # clamp
        assert fixed[2] == "0 0.3 0.3 0.05 0.05\n"   # 完好行原样保留

    def test_clean_file_untouched(self, tmp_path):
        p = tmp_path / "ok.txt"
        p.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        fixed, stats = repair_label_file(str(p), nc=1)
        assert fixed is None
        assert all(v == 0 for v in stats.values())

    def test_empty_label_file_valid(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        fixed, _ = repair_label_file(str(p), nc=1)
        assert fixed is None

    def test_multiclass_oob_dropped_not_remapped(self, tmp_path):
        p = tmp_path / "mc.txt"
        p.write_text("7 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        fixed, stats = repair_label_file(str(p), nc=3)
        assert stats["rows_dropped"] == 1
        assert stats["class_id_fixed"] == 0
        assert len(fixed) == 1

    def test_full_run_backs_up_and_rewrites(self, tmp_path):
        root, yaml_path = make_dataset(tmp_path)
        report = repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))

        labels = report["labels"]
        assert labels["files_changed"] == 1
        assert labels["class_id_fixed"] == 1
        assert labels["rows_dropped"] == 2

        fixed_text = (root / "labels" / "train" / "bad.txt").read_text(encoding="utf-8")
        assert fixed_text.splitlines()[0].startswith("0 ")
        assert "garbage" not in fixed_text
        backup = root / "_quarantine" / "label_backup" / "train" / "bad.txt"
        assert backup.read_text(encoding="utf-8").startswith("3 ")

    def test_unlabeled_split_reported_not_fixed(self, tmp_path):
        root, yaml_path = make_dataset(tmp_path)
        report = repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))
        assert report["labels"]["unlabeled_splits"] == {"test": 2}

    def test_caches_cleared_when_changed(self, tmp_path):
        root, yaml_path = make_dataset(tmp_path)
        report = repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))
        assert report["changed"] is True
        assert not (root / "labels" / "train.cache").exists()
        assert any(p.endswith("train.cache") for p in report["caches_removed"])

    def test_no_changes_keeps_caches(self, tmp_path):
        root, yaml_path = make_dataset(tmp_path, with_dups=False,
                                       with_bad_labels=False,
                                       unlabeled_test=False)
        (root / "labels" / "train.cache").write_bytes(b"cache")
        report = repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))
        assert report["changed"] is False
        assert (root / "labels" / "train.cache").exists()

    def test_report_json_written(self, tmp_path):
        _, yaml_path = make_dataset(tmp_path)
        report = repair_dataset(yaml_path, reports_dir=str(tmp_path / "reports"))
        with open(report["report_json"], encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["duplicates"]["identical_quarantined"] == 1


# ════════════════════════════════════════════════════════════════════
# Orchestrator 接线 — 出错不阻塞, 正常发事件
# ════════════════════════════════════════════════════════════════════

class FakeEvents:
    def __init__(self):
        self.emitted = []

    def emit(self, event_type, **data):
        self.emitted.append((event_type, data))


def make_orch(**attrs):
    o = object.__new__(Orchestrator)
    o.events = FakeEvents()
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


class TestOrchestratorWiring:
    def test_repair_runs_and_emits(self, tmp_path, monkeypatch):
        _, yaml_path = make_dataset(tmp_path)
        o = make_orch(data_yaml=yaml_path, reports_dir=str(tmp_path / "reports"))
        o._run_dataset_repair()

        events = dict(o.events.emitted)
        assert "dataset_repair" in events
        assert events["dataset_repair"]["duplicates_quarantined"] == 1
        assert events["dataset_repair"]["unlabeled_splits"] == {"test": 2}

    def test_repair_failure_never_blocks(self, monkeypatch):
        o = make_orch(data_yaml="/nonexistent/person.yaml", reports_dir="/tmp/x")

        def boom(*a, **kw):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(dataset_repair, "repair_dataset", boom)
        o._run_dataset_repair()  # 不应抛出
        assert o.events.emitted == []

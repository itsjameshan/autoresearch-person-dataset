"""test_training_config_fixes.py — 训练失败根因修复的回归测试

针对 dashboard 上"实验全失败 / CDS 上不去"的三个仓库内根因:
  1. model.train() 硬编码 imgsz=640 — IMGSZ 旋钮 (Researcher/HPO/人工)
     全部失效, small_obj_recall ≈ 0 (CDS 的 15% 归零)
  2. 没有墙钟上限 — yolo12x@batch2 在 5070 上 32 epochs 远超调度器
     2h 看门狗 → 每次被 SIGKILL 记为 failed; 且 kill 信息不进 run.log,
     root_cause 永远是 unknown
  3. person.yaml 相对 `path:` 由 ultralytics 按 datasets_dir 解析,
     跨机器漂移 (历史上 3 个 commit 修同一类问题); 模型 fallback 列表
     漏 yolo12x 且按字母序选"最大" (s 排在 l 后面)

运行: python -m pytest test_training_config_fixes.py -v
"""

import os
import re
import shutil
import time

import pytest
import yaml

from train_utils import resolve_data_yaml, resolve_model, MODEL_PREFERENCE
from autoresearch_v2.tools import train_dispatcher, train_guard

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(REPO_ROOT, "train.py")


# ════════════════════════════════════════════════════════════════════
# resolve_data_yaml — 图片路径跨机器解析
# ════════════════════════════════════════════════════════════════════

def _write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


class TestResolveDataYaml:
    def test_relative_path_resolved_to_yaml_dir(self, tmp_path):
        # 实际布局: person_dataset/person.yaml + path: person_dataset
        # (相对) + images 就在 yaml 旁边
        ds = tmp_path / "person_dataset"
        (ds / "images" / "train").mkdir(parents=True)
        yaml_path = ds / "person.yaml"
        _write_yaml(str(yaml_path), {
            "path": "person_dataset", "train": "images/train",
            "val": "images/val", "nc": 1, "names": {0: "person"},
        })
        out = resolve_data_yaml(str(yaml_path), str(tmp_path / "runs"))

        assert out != str(yaml_path), "应返回解析副本"
        resolved = yaml.safe_load(open(out, encoding="utf-8"))
        assert os.path.isabs(resolved["path"])
        assert os.path.samefile(resolved["path"], str(ds))
        assert resolved["train"] == "images/train"

    def test_relative_path_sibling_dir(self, tmp_path):
        # yaml 在仓库根, path: person_dataset 指向子目录
        (tmp_path / "person_dataset" / "images" / "train").mkdir(parents=True)
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(str(yaml_path), {"path": "person_dataset",
                                     "train": "images/train"})
        out = resolve_data_yaml(str(yaml_path), str(tmp_path / "runs"))
        resolved = yaml.safe_load(open(out, encoding="utf-8"))
        assert os.path.samefile(resolved["path"],
                                str(tmp_path / "person_dataset"))

    def test_valid_absolute_path_untouched(self, tmp_path):
        ds = tmp_path / "ds"
        (ds / "images" / "train").mkdir(parents=True)
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(str(yaml_path), {"path": str(ds), "train": "images/train"})
        out = resolve_data_yaml(str(yaml_path), str(tmp_path / "runs"))
        assert out == str(yaml_path), "有效绝对路径不应被改写"

    def test_stale_absolute_path_re_resolved(self, tmp_path):
        # 机器迁移后 yaml 里残留旧机器的绝对路径 — 按 yaml 位置重新解析
        ds = tmp_path / "person_dataset"
        (ds / "images" / "train").mkdir(parents=True)
        yaml_path = ds / "person.yaml"
        _write_yaml(str(yaml_path), {"path": "D:\\old_machine\\person_dataset"
                                     if os.name == "nt" else "/old/machine/ds",
                                     "train": "images/train"})
        out = resolve_data_yaml(str(yaml_path), str(tmp_path / "runs"))
        resolved = yaml.safe_load(open(out, encoding="utf-8"))
        assert os.path.samefile(resolved["path"], str(ds))

    def test_unresolvable_returns_original(self, tmp_path):
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(str(yaml_path), {"path": "nowhere", "train": "images/train"})
        out = resolve_data_yaml(str(yaml_path), str(tmp_path / "runs"))
        assert out == str(yaml_path)

    def test_garbage_yaml_returns_original(self, tmp_path):
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(":\nnot valid: [yaml", encoding="utf-8")
        out = resolve_data_yaml(str(yaml_path), str(tmp_path / "runs"))
        assert out == str(yaml_path)


# ════════════════════════════════════════════════════════════════════
# resolve_model — 容量优先级 fallback
# ════════════════════════════════════════════════════════════════════

class TestResolveModel:
    def test_requested_model_used_when_present(self, tmp_path):
        (tmp_path / "yolo12l.pt").write_bytes(b"w")
        got = resolve_model("yolo12l.pt", str(tmp_path))
        assert os.path.samefile(got, str(tmp_path / "yolo12l.pt"))

    def test_fallback_prefers_capacity_not_alphabet(self, tmp_path):
        # 旧 bug: sorted()[-1] 按字母序 → yolo12s 排在 yolo12m 后面被选中
        (tmp_path / "yolo12s.pt").write_bytes(b"w")
        (tmp_path / "yolo12m.pt").write_bytes(b"w")
        got = resolve_model("yolo12l.pt", str(tmp_path))
        assert got.endswith("yolo12m.pt"), "应按容量优先 (m > s), 不是字母序"

    def test_fallback_includes_yolo12x(self, tmp_path):
        # 旧 fallback 列表漏掉 x — 本机最大的模型反而永远不被选
        (tmp_path / "yolo12x.pt").write_bytes(b"w")
        (tmp_path / "yolo12n.pt").write_bytes(b"w")
        got = resolve_model("yolo12l.pt", str(tmp_path))
        assert got.endswith("yolo12x.pt")

    def test_no_models_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            resolve_model("yolo12l.pt", str(tmp_path))

    def test_preference_order_sane(self):
        assert MODEL_PREFERENCE[0] == "yolo12x.pt"
        assert MODEL_PREFERENCE[-1] == "yolo12n.pt"


# ════════════════════════════════════════════════════════════════════
# 看门狗超时 → root_cause=training_timeout (不再是 unknown)
# ════════════════════════════════════════════════════════════════════

class TestTimeoutAttribution:
    def test_watchdog_kill_marks_timeout_in_log_tail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: [])
        monkeypatch.setattr(train_guard, "write_pid_file",
                            lambda *a, **kw: None)
        monkeypatch.setattr(train_guard, "clear_pid_file",
                            lambda *a, **kw: None)
        monkeypatch.setattr(train_dispatcher, "_check_gpu_or_raise",
                            lambda: None)

        class SlowProc:
            pid = 999
            returncode = 0

            def __init__(self):
                self.killed = False

                def lines():
                    time.sleep(0.7)   # 看门狗 (0.15s) 先到
                    yield "Epoch 1/32 ...\n"

                self.stdout = lines()

            def poll(self):
                return 0 if self.killed else None

            def kill(self):
                self.killed = True

            def wait(self):
                return 0

        monkeypatch.setattr(train_dispatcher.subprocess, "Popen",
                            lambda *a, **kw: SlowProc())

        ok, tail = train_dispatcher.run_training(
            log_file=str(tmp_path / "run.log"), timeout=0.15, quiet=True)

        assert ok is False
        assert "timed_out" in tail
        assert train_dispatcher._detect_root_cause(tail) == "training_timeout"


# ════════════════════════════════════════════════════════════════════
# train.py 配置接线 — 源码级回归守卫 (本环境无 ultralytics, 不能 import)
# ════════════════════════════════════════════════════════════════════

class TestTrainPyWiring:
    @pytest.fixture
    def src(self):
        with open(TRAIN_PY, "r", encoding="utf-8") as f:
            return f.read()

    def test_imgsz_uses_config_variable(self, src):
        assert re.search(r"imgsz\s*=\s*IMGSZ\s*,", src), \
            "model.train 必须用 imgsz=IMGSZ"
        assert not re.search(r"imgsz\s*=\s*\d+\s*,", src), \
            "model.train 里不允许再硬编码 imgsz 数字"

    def test_time_limit_wired(self, src):
        assert re.search(r"time\s*=\s*TIME_LIMIT_H\s*,", src), \
            "model.train 必须带墙钟上限 time=TIME_LIMIT_H"

    def test_config_values_fit_dispatcher_watchdog(self):
        cfg = train_dispatcher.read_current_config(TRAIN_PY)
        assert "TIME_LIMIT_H" in cfg
        # 训练时限 + 收尾评估必须在调度器看门狗之内, 否则又回到 SIGKILL
        assert cfg["TIME_LIMIT_H"] * 3600 < train_dispatcher.TRAIN_TIMEOUT
        assert cfg["TIME_LIMIT_H"] * 3600 <= train_dispatcher.TRAIN_TIMEOUT - 600, \
            "至少留 10 分钟余量给模型加载/缓存/评估"
        assert isinstance(cfg["BATCH"], int) and cfg["BATCH"] >= 1
        assert str(cfg["MODEL"]).endswith(".pt")
        assert cfg["IMGSZ"] >= 640

    def test_imgsz_knob_round_trips_via_config_diff(self, tmp_path):
        # 端到端: Researcher/HPO 改 IMGSZ → 配置真的变 (以前改了也没用)
        copy = str(tmp_path / "train_copy.py")
        shutil.copyfile(TRAIN_PY, copy)
        changed = train_dispatcher.apply_config_diff({"IMGSZ": 1280}, copy)
        assert changed
        assert train_dispatcher.read_current_config(copy)["IMGSZ"] == 1280
        src = open(copy, encoding="utf-8").read()
        assert re.search(r"imgsz\s*=\s*IMGSZ\s*,", src), \
            "config_diff 不应破坏 model.train 的 imgsz=IMGSZ 接线"

"""test_runaway_training_fixes.py — 并发 train.py / CDS=0.0000 假完成行 修复验证

覆盖 5 项修复 (无 GPU, 全部用 mock):
  1. 单例训练守卫: train_guard 探测/等待/收割孤儿 + run_training spawn 闸门
     + agent_overseer 停止时级联杀子进程树 / 重启前闸门
  2. fail-fast 记录: 评估空指标 → status=failed + 跳过 git 自动提交;
     optuna 不把空 metrics 当 completed 入库
  3. HPO 断路器: 判据从 n_completed 改为「产出非空 metrics 的 trial 数」
  4. 监督者违规滑动窗口 + 恢复重置 (不再永久建议紧急停止)

运行: python -m pytest test_stuck_detector.py test_runaway_training_fixes.py -v
"""

import json
import os
import time
import types
from datetime import datetime

import pytest

from autoresearch_v2.tools import train_guard
from autoresearch_v2.tools import train_dispatcher
from autoresearch_v2.tools.optuna_runner import OptunaRunner, OptunaSweepResult
from autoresearch_v2.orchestrator import Orchestrator
import agent_overseer
from supervisor_agent import SafetyGuard


PYEXE = "C:\\Python311\\python.exe"
TRAIN_ABS = "C:\\proj\\person_dataset\\train.py"


# ════════════════════════════════════════════════════════════════════
# 假 psutil / 假进程
# ════════════════════════════════════════════════════════════════════

class FakeProc:
    def __init__(self, pid, cmdline=None, create_time=0.0, parent=None,
                 running=True, children=None):
        self.pid = pid
        self._cmdline = list(cmdline or [])
        self._create = float(create_time)
        self._parent = parent
        self._running = running
        self._children = list(children or [])
        self.terminated = False
        self.killed = False

    @property
    def info(self):
        return {"pid": self.pid, "cmdline": self._cmdline,
                "create_time": self._create}

    def create_time(self):
        return self._create

    def parent(self):
        return self._parent

    def is_running(self):
        return self._running and not self.terminated and not self.killed

    def terminate(self):
        self.terminated = True
        self._running = False

    def kill(self):
        self.killed = True
        self._running = False

    def wait(self, timeout=None):
        return 0

    def children(self, recursive=False):
        return list(self._children)


class FakePsutil:
    NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    AccessDenied = type("AccessDenied", (Exception,), {})
    TimeoutExpired = type("TimeoutExpired", (Exception,), {})

    def __init__(self, procs=()):
        self.procs = list(procs)

    def process_iter(self, attrs=None):
        return iter(self.procs)

    def Process(self, pid):
        for p in self.procs:
            if p.pid == pid:
                return p
        raise self.NoSuchProcess(pid)

    def wait_procs(self, procs, timeout=None):
        gone = [p for p in procs if not p.is_running()]
        alive = [p for p in procs if p.is_running()]
        return gone, alive

    def pid_exists(self, pid):
        return any(p.pid == pid for p in self.procs)


def trainer_proc(pid, **kw):
    return FakeProc(pid, cmdline=[PYEXE, "-u", TRAIN_ABS], **kw)


# ════════════════════════════════════════════════════════════════════
# 假 orchestrator 依赖
# ════════════════════════════════════════════════════════════════════

class FakeState:
    def __init__(self):
        self.inserted = []
        self.status_updates = []
        self.decisions = []

    def insert_experiment(self, **kw):
        self.inserted.append(kw)
        return kw.get("run_id")

    def update_experiment_status(self, run_id, status, gpu_minutes=0.0,
                                 cost_usd=0.0, root_cause=""):
        self.status_updates.append({
            "run_id": run_id, "status": status,
            "gpu_minutes": gpu_minutes, "root_cause": root_cause,
        })

    def log_decision(self, **kw):
        self.decisions.append(kw)


class FakeEvents:
    def __init__(self):
        self.emitted = []

    def emit(self, event_type, **data):
        self.emitted.append((event_type, data))

    def types(self):
        return [e[0] for e in self.emitted]


class FakeBudget:
    def __init__(self):
        self.released = []

    def release_reserved(self, run_id):
        self.released.append(run_id)


def make_orchestrator(**attrs):
    o = object.__new__(Orchestrator)
    o.state = FakeState()
    o.events = FakeEvents()
    o.budget = FakeBudget()
    o.consecutive_fails = 0
    o.max_consecutive_fails = 3
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


# ════════════════════════════════════════════════════════════════════
# 修复1a — train_guard: cmdline 匹配
# ════════════════════════════════════════════════════════════════════

class TestCmdlineMatch:
    def test_matches_abs_path(self):
        assert train_guard._cmdline_is_trainer([PYEXE, "-u", TRAIN_ABS])

    def test_matches_bare_name(self):
        assert train_guard._cmdline_is_trainer(["python", "train.py"])

    def test_case_insensitive_windows(self):
        assert train_guard._cmdline_is_trainer(["python", "C:\\X\\TRAIN.PY"])

    def test_orchestrator_not_matched(self):
        assert not train_guard._cmdline_is_trainer(
            ["python", "-m", "autoresearch_v2.orchestrator"])

    def test_supervisor_not_matched(self):
        assert not train_guard._cmdline_is_trainer(
            ["python", "-u", "supervisor_agent.py"])

    def test_similar_names_not_matched(self):
        assert not train_guard._cmdline_is_trainer(["python", "retrain.py"])
        assert not train_guard._cmdline_is_trainer(["python", "my_train.pyc"])

    def test_empty_and_none(self):
        assert not train_guard._cmdline_is_trainer([])
        assert not train_guard._cmdline_is_trainer(None)


# ════════════════════════════════════════════════════════════════════
# 修复1b — train_guard: 存活 trainer 探测
# ════════════════════════════════════════════════════════════════════

class TestFindLiveTrainers:
    def test_finds_trainers_only(self, monkeypatch):
        fake = FakePsutil([
            trainer_proc(101),
            FakeProc(102, cmdline=["python", "-m", "autoresearch_v2.orchestrator"]),
            trainer_proc(103),
        ])
        monkeypatch.setattr(train_guard, "psutil", fake)
        got = train_guard.find_live_trainers()
        assert sorted(t["pid"] for t in got) == [101, 103]

    def test_exclude_pids(self, monkeypatch):
        fake = FakePsutil([trainer_proc(101), trainer_proc(103)])
        monkeypatch.setattr(train_guard, "psutil", fake)
        got = train_guard.find_live_trainers(exclude_pids=(103,))
        assert [t["pid"] for t in got] == [101]

    def test_excludes_self(self, monkeypatch):
        fake = FakePsutil([trainer_proc(os.getpid())])
        monkeypatch.setattr(train_guard, "psutil", fake)
        assert train_guard.find_live_trainers() == []

    def test_pid_file_fallback_alive(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_guard, "psutil", None)
        pid_file = str(tmp_path / "train.pid")
        train_guard.write_pid_file(4242, TRAIN_ABS, path=pid_file)
        monkeypatch.setattr(train_guard, "_pid_alive_no_psutil", lambda pid: True)
        got = train_guard.find_live_trainers(pid_file=pid_file)
        assert len(got) == 1 and got[0]["pid"] == 4242

    def test_pid_file_fallback_dead(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_guard, "psutil", None)
        pid_file = str(tmp_path / "train.pid")
        train_guard.write_pid_file(4242, TRAIN_ABS, path=pid_file)
        monkeypatch.setattr(train_guard, "_pid_alive_no_psutil", lambda pid: False)
        assert train_guard.find_live_trainers(pid_file=pid_file) == []

    def test_pid_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_guard, "psutil", None)
        assert train_guard.find_live_trainers(
            pid_file=str(tmp_path / "nope.pid")) == []


class TestPidFile:
    def test_roundtrip(self, tmp_path):
        p = str(tmp_path / "train.pid")
        train_guard.write_pid_file(777, TRAIN_ABS, path=p)
        rec = train_guard.read_pid_file(p)
        assert rec["pid"] == 777
        assert rec["train_script"] == TRAIN_ABS
        assert rec["created"] > 0

    def test_clear_matching_pid(self, tmp_path):
        p = str(tmp_path / "train.pid")
        train_guard.write_pid_file(777, path=p)
        train_guard.clear_pid_file(777, path=p)
        assert not os.path.exists(p)

    def test_clear_keeps_foreign_pid(self, tmp_path):
        # 另一个 trainer 刚写入自己的 pid — 我们的清理不能误删它的记录
        p = str(tmp_path / "train.pid")
        train_guard.write_pid_file(888, path=p)
        train_guard.clear_pid_file(777, path=p)
        assert os.path.exists(p)
        assert train_guard.read_pid_file(p)["pid"] == 888

    def test_clear_unconditional(self, tmp_path):
        p = str(tmp_path / "train.pid")
        train_guard.write_pid_file(888, path=p)
        train_guard.clear_pid_file(path=p)
        assert not os.path.exists(p)

    def test_read_garbage_returns_none(self, tmp_path):
        p = tmp_path / "train.pid"
        p.write_text("not json at all")
        assert train_guard.read_pid_file(str(p)) is None


class TestWaitForTrainers:
    def test_immediate_when_clear(self, monkeypatch):
        monkeypatch.setattr(train_guard, "find_live_trainers",
                            lambda **kw: [])
        assert train_guard.wait_for_trainers_to_exit(10.0) == []

    def test_timeout_returns_survivors(self, monkeypatch):
        live = [{"pid": 1, "cmdline": [], "create_time": 0.0}]
        monkeypatch.setattr(train_guard, "find_live_trainers",
                            lambda **kw: list(live))
        got = train_guard.wait_for_trainers_to_exit(0.0)
        assert [t["pid"] for t in got] == [1]

    def test_clears_after_polls(self, monkeypatch):
        seq = [[{"pid": 1}], [{"pid": 1}], []]
        monkeypatch.setattr(train_guard, "find_live_trainers",
                            lambda **kw: seq.pop(0))
        got = train_guard.wait_for_trainers_to_exit(5.0, poll_sec=0.01)
        assert got == []


# ════════════════════════════════════════════════════════════════════
# 修复1c — train_guard: 孤儿收割 (只杀 早于启动 + 无存活父进程)
# ════════════════════════════════════════════════════════════════════

class TestReapOrphans:
    def test_reaps_only_parentless_and_older(self, monkeypatch, tmp_path):
        now = time.time()
        live_parent = FakeProc(900, cmdline=["python", "-m", "autoresearch_v2.orchestrator"])
        dead_parent = FakeProc(901, cmdline=["python"], running=False)
        orphan = trainer_proc(201, create_time=now - 1000, parent=None)
        orphan_dead_parent = trainer_proc(202, create_time=now - 1000, parent=dead_parent)
        managed = trainer_proc(203, create_time=now - 1000, parent=live_parent)
        younger = trainer_proc(204, create_time=now + 100, parent=None)
        fake = FakePsutil([orphan, orphan_dead_parent, managed, younger,
                           live_parent, dead_parent])
        monkeypatch.setattr(train_guard, "psutil", fake)

        reaped = train_guard.reap_orphan_trainers(
            started_before=now, pid_file=str(tmp_path / "train.pid"))

        assert sorted(r["pid"] for r in reaped) == [201, 202]
        assert orphan.terminated and orphan_dead_parent.terminated
        assert not managed.terminated, "有存活父进程的 trainer 不能被抢杀"
        assert not younger.terminated, "晚于自身启动的 trainer 不能被收割"

    def test_init_adopted_is_orphan(self, monkeypatch, tmp_path):
        # POSIX 上孤儿被 init(pid=1) 收养 — 仍视为孤儿
        now = time.time()
        init = FakeProc(1, cmdline=["init"])
        adopted = trainer_proc(205, create_time=now - 50, parent=init)
        fake = FakePsutil([adopted, init])
        monkeypatch.setattr(train_guard, "psutil", fake)
        reaped = train_guard.reap_orphan_trainers(
            started_before=now, pid_file=str(tmp_path / "train.pid"))
        assert [r["pid"] for r in reaped] == [205]

    def test_no_psutil_no_reaping(self, monkeypatch):
        monkeypatch.setattr(train_guard, "psutil", None)
        assert train_guard.reap_orphan_trainers(started_before=time.time()) == []

    def test_clears_stale_pid_file(self, monkeypatch, tmp_path):
        pid_file = str(tmp_path / "train.pid")
        train_guard.write_pid_file(99999, path=pid_file)
        monkeypatch.setattr(train_guard, "psutil", FakePsutil([]))
        train_guard.reap_orphan_trainers(started_before=time.time(),
                                         pid_file=pid_file)
        assert not os.path.exists(pid_file), "指向死进程的陈旧 PID 文件应被清理"


class TestTerminateTree:
    def test_kills_children_not_parent(self, monkeypatch):
        c1, c2 = trainer_proc(301), trainer_proc(302)
        parent = FakeProc(300, cmdline=["python"], children=[c1, c2])
        fake = FakePsutil([parent, c1, c2])
        monkeypatch.setattr(train_guard, "psutil", fake)
        killed = train_guard.terminate_tree(300, include_parent=False)
        assert sorted(killed) == [301, 302]
        assert c1.terminated and c2.terminated
        assert not parent.terminated

    def test_include_parent(self, monkeypatch):
        c1 = trainer_proc(311)
        parent = FakeProc(310, cmdline=["python"], children=[c1])
        fake = FakePsutil([parent, c1])
        monkeypatch.setattr(train_guard, "psutil", fake)
        killed = train_guard.terminate_tree(310, include_parent=True)
        assert sorted(killed) == [310, 311]

    def test_no_psutil(self, monkeypatch):
        monkeypatch.setattr(train_guard, "psutil", None)
        assert train_guard.terminate_tree(123) == []


# ════════════════════════════════════════════════════════════════════
# 修复1d — run_training spawn 闸门 + PID 文件生命周期
# ════════════════════════════════════════════════════════════════════

class FakePopen:
    def __init__(self, lines=("epoch 1/10 done\n",), returncode=0, pid=777):
        self.pid = pid
        self.stdout = iter(lines)
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        pass


class TestRunTrainingSingleton:
    def test_refuses_to_spawn_when_trainer_alive(self, monkeypatch):
        blocker = [{"pid": 4242, "cmdline": [], "create_time": 0.0}]
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: list(blocker))
        monkeypatch.setattr(train_guard, "wait_for_trainers_to_exit",
                            lambda *a, **kw: list(blocker))
        spawned = []
        monkeypatch.setattr(train_dispatcher.subprocess, "Popen",
                            lambda *a, **kw: spawned.append(1))
        gpu_checked = []
        monkeypatch.setattr(train_dispatcher, "_check_gpu_or_raise",
                            lambda: gpu_checked.append(1))

        ok, tail = train_dispatcher.run_training(
            quiet=True, singleton_wait_sec=0.0)

        assert ok is False
        assert "refusing to spawn" in tail
        assert "4242" in tail
        assert spawned == [], "已有存活 trainer 时绝不能再 spawn"
        assert gpu_checked == []

    def test_spawns_after_blocker_exits(self, monkeypatch, tmp_path):
        blocker = [{"pid": 4242, "cmdline": [], "create_time": 0.0}]
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: list(blocker))
        monkeypatch.setattr(train_guard, "wait_for_trainers_to_exit",
                            lambda *a, **kw: [])  # 等待期间退出了
        monkeypatch.setattr(train_dispatcher, "_check_gpu_or_raise", lambda: None)
        monkeypatch.setattr(train_dispatcher.subprocess, "Popen",
                            lambda *a, **kw: FakePopen())
        pid_ops = []
        monkeypatch.setattr(train_guard, "write_pid_file",
                            lambda pid, train_script="", path=None: pid_ops.append(("write", pid)))
        monkeypatch.setattr(train_guard, "clear_pid_file",
                            lambda pid=None, path=None: pid_ops.append(("clear", pid)))

        ok, tail = train_dispatcher.run_training(
            log_file=str(tmp_path / "run.log"), timeout=60,
            quiet=True, singleton_wait_sec=5.0)

        assert ok is True
        assert "epoch 1/10" in tail

    def test_pid_file_written_and_cleared(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: [])
        monkeypatch.setattr(train_dispatcher, "_check_gpu_or_raise", lambda: None)
        monkeypatch.setattr(train_dispatcher.subprocess, "Popen",
                            lambda *a, **kw: FakePopen(pid=777))
        pid_ops = []
        monkeypatch.setattr(train_guard, "write_pid_file",
                            lambda pid, train_script="", path=None: pid_ops.append(("write", pid)))
        monkeypatch.setattr(train_guard, "clear_pid_file",
                            lambda pid=None, path=None: pid_ops.append(("clear", pid)))

        ok, _ = train_dispatcher.run_training(
            log_file=str(tmp_path / "run.log"), timeout=60, quiet=True)

        assert ok is True
        assert pid_ops == [("write", 777), ("clear", 777)]

    def test_pid_file_cleared_even_when_tee_crashes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: [])
        monkeypatch.setattr(train_dispatcher, "_check_gpu_or_raise", lambda: None)

        class ExplodingStdout:
            def __iter__(self):
                raise RuntimeError("boom")

        def make_proc(*a, **kw):
            p = FakePopen(pid=778)
            p.stdout = ExplodingStdout()
            return p

        monkeypatch.setattr(train_dispatcher.subprocess, "Popen", make_proc)
        pid_ops = []
        monkeypatch.setattr(train_guard, "write_pid_file",
                            lambda pid, train_script="", path=None: pid_ops.append(("write", pid)))
        monkeypatch.setattr(train_guard, "clear_pid_file",
                            lambda pid=None, path=None: pid_ops.append(("clear", pid)))

        with pytest.raises(RuntimeError):
            train_dispatcher.run_training(
                log_file=str(tmp_path / "run.log"), timeout=60, quiet=True)
        assert ("clear", 778) in pid_ops, "异常路径也必须清理 PID 文件"

    def test_root_cause_concurrent_training(self):
        tail = ("concurrent train.py still alive (pid=[4242]) after 300s "
                "— refusing to spawn a second trainer")
        assert train_dispatcher._detect_root_cause(tail) == "concurrent_training"

    def test_dispatch_records_concurrent_failure(self, monkeypatch, tmp_path):
        st = FakeState()
        d = train_dispatcher.TrainDispatcher(
            state=st,
            train_script=str(tmp_path / "train.py"),
            log_file=str(tmp_path / "run.log"),
        )
        monkeypatch.setattr(
            train_dispatcher, "run_training",
            lambda *a, **kw: (False, "concurrent train.py still alive (pid=[1]) "
                                     "— refusing to spawn a second trainer"))
        with pytest.raises(train_dispatcher.TrainingFailed):
            d.dispatch("run_cc", {"config_diff": {}})
        assert st.status_updates[-1]["status"] == "failed"
        assert st.status_updates[-1]["root_cause"] == "concurrent_training"


# ════════════════════════════════════════════════════════════════════
# 修复1e — agent_overseer: 停止级联杀子树 / 启动重启闸门
# ════════════════════════════════════════════════════════════════════

class StubAgent:
    """ManagedAgent 替身 — 只记录 launch/stop 调用, 不 spawn 真进程。"""
    def __init__(self, name, alive=False):
        self.name = name
        self._alive = alive
        self.launched = 0
        self.stopped = 0
        self.status = "running" if alive else "stopped"

    def is_alive(self):
        return self._alive

    def launch(self):
        self.launched += 1
        self._alive = True
        return True

    def stop(self, timeout=15):
        self.stopped += 1
        self._alive = False


def make_overseer(*agents):
    ov = object.__new__(agent_overseer.AgentOverseer)
    ov.agents = list(agents)
    return ov


@pytest.fixture
def fast_sleep(monkeypatch):
    monkeypatch.setattr(agent_overseer.time, "sleep", lambda s: None)


class TestOverseerTrainerGate:
    def test_restart_blocked_while_trainer_alive(self, monkeypatch, fast_sleep):
        stub = StubAgent("Orchestrator")
        ov = make_overseer(stub)
        live = [{"pid": 4242, "cmdline": [], "create_time": 0.0}]
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: list(live))
        monkeypatch.setattr(train_guard, "wait_for_trainers_to_exit",
                            lambda *a, **kw: list(live))

        assert ov.restart_agent("Orchestrator") is False
        assert stub.launched == 0, "存活 trainer 未退出时不能重启 Orchestrator"

    def test_restart_proceeds_after_trainer_exits(self, monkeypatch, fast_sleep):
        stub = StubAgent("Orchestrator", alive=True)
        ov = make_overseer(stub)
        monkeypatch.setattr(train_guard, "find_live_trainers",
                            lambda **kw: [{"pid": 4242}])
        monkeypatch.setattr(train_guard, "wait_for_trainers_to_exit",
                            lambda *a, **kw: [])

        assert ov.restart_agent("Orchestrator") is True
        assert stub.stopped == 1 and stub.launched == 1

    def test_restart_no_trainers_immediate(self, monkeypatch, fast_sleep):
        stub = StubAgent("Orchestrator")
        ov = make_overseer(stub)
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: [])
        assert ov.restart_agent("Orchestrator") is True
        assert stub.launched == 1

    def test_non_trainer_agent_not_gated(self, monkeypatch, fast_sleep):
        stub = StubAgent("Dashboard-v2")
        ov = make_overseer(stub)

        def boom(**kw):
            raise AssertionError("非训练类 agent 不应触发 trainer 闸门")

        monkeypatch.setattr(train_guard, "find_live_trainers", boom)
        assert ov.restart_agent("Dashboard-v2") is True
        assert stub.launched == 1

    def test_start_agent_blocked_while_trainer_alive(self, monkeypatch, fast_sleep):
        stub = StubAgent("Orchestrator")
        ov = make_overseer(stub)
        live = [{"pid": 1}]
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: list(live))
        monkeypatch.setattr(train_guard, "wait_for_trainers_to_exit",
                            lambda *a, **kw: list(live))
        assert ov.start_agent("Orchestrator") is False
        assert stub.launched == 0

    def test_unknown_agent(self, fast_sleep):
        ov = make_overseer()
        assert ov.restart_agent("Nobody") is False


class TestManagedAgentStopKillsTree:
    def test_stop_terminates_children_first(self, monkeypatch):
        a = agent_overseer.ManagedAgent("Orchestrator", "role", ["python", "x"])

        class P:
            pid = 555

            def __init__(self):
                self.terminated = False

            def poll(self):
                return None if not self.terminated else 0

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.terminated = True

        a.process = P()
        tree_calls = []
        monkeypatch.setattr(
            train_guard, "terminate_tree",
            lambda pid, include_parent=False, grace_sec=10.0:
                tree_calls.append((pid, include_parent)) or [601])

        a.stop()

        assert tree_calls == [(555, False)], "停止前必须级联终止子进程树"
        assert a.process.terminated
        assert a.status == "stopped"


# ════════════════════════════════════════════════════════════════════
# 修复1f — Orchestrator 启动收割孤儿
# ════════════════════════════════════════════════════════════════════

class TestOrchestratorReap:
    def test_reap_uses_own_start_time_and_emits(self, monkeypatch):
        o = make_orchestrator(_start_time=12345.0)
        seen = {}
        monkeypatch.setattr(
            train_guard, "reap_orphan_trainers",
            lambda started_before: seen.update(sb=started_before) or [{"pid": 11}])
        monkeypatch.setattr(train_guard, "find_live_trainers",
                            lambda **kw: [{"pid": 22}])

        o._reap_orphan_trainers()

        assert seen["sb"] == 12345.0
        assert ("orphan_trainers_reaped", {"pids": [11]}) in o.events.emitted
        assert ("live_trainers_detected", {"pids": [22]}) in o.events.emitted

    def test_reap_quiet_when_nothing_found(self, monkeypatch):
        o = make_orchestrator(_start_time=1.0)
        monkeypatch.setattr(train_guard, "reap_orphan_trainers",
                            lambda started_before: [])
        monkeypatch.setattr(train_guard, "find_live_trainers", lambda **kw: [])
        o._reap_orphan_trainers()
        assert o.events.emitted == []

    def test_reap_survives_guard_exception(self, monkeypatch):
        o = make_orchestrator(_start_time=1.0)

        def boom(started_before):
            raise RuntimeError("psutil exploded")

        monkeypatch.setattr(train_guard, "reap_orphan_trainers", boom)
        o._reap_orphan_trainers()  # 不应抛出


# ════════════════════════════════════════════════════════════════════
# 修复2 — fail-fast 记录: 空评估 → failed, 跳过 git 提交
# ════════════════════════════════════════════════════════════════════

class TestRecordEvalFailure:
    def test_marks_failed_counts_and_releases(self):
        o = make_orchestrator()
        halt = o._record_eval_failure("run_x", "eval_empty_metrics")

        assert halt is False
        assert o.consecutive_fails == 1
        assert o.state.status_updates == [{
            "run_id": "run_x", "status": "failed",
            "gpu_minutes": 0.0, "root_cause": "eval_empty_metrics",
        }]
        assert o.budget.released == ["run_x"]
        assert ("eval_failed", {"run_id": "run_x", "reason": "eval_empty_metrics"}) \
            in o.events.emitted

    def test_halts_at_max_consecutive_fails(self):
        o = make_orchestrator()
        assert o._record_eval_failure("r1", "eval_empty_metrics") is False
        assert o._record_eval_failure("r2", "eval_empty_metrics") is False
        assert o._record_eval_failure("r3", "eval_empty_metrics") is True
        assert o.consecutive_fails == 3

    def test_state_error_does_not_mask_failure_count(self):
        o = make_orchestrator()

        def boom(*a, **kw):
            raise RuntimeError("db locked")

        o.state.update_experiment_status = boom
        halt = o._record_eval_failure("run_x", "eval_empty_metrics")
        assert halt is False
        assert o.consecutive_fails == 1


class TestGitCommitGuard:
    def _patch_git(self, monkeypatch):
        calls = []

        def fake_run(argv, **kw):
            calls.append(list(argv))
            return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

        import subprocess as _sp
        monkeypatch.setattr(_sp, "run", fake_run)
        return calls

    def test_skips_empty_metrics(self, monkeypatch):
        calls = self._patch_git(monkeypatch)
        o = make_orchestrator()
        o._git_commit("run_x", 1, "TRAIN", metrics={})
        o._git_commit("run_x", 1, "TRAIN", metrics=None)
        o._git_commit("run_x", 1, "HPO", metrics={"precision": 0.5})  # 无 cds 键
        assert calls == [], "失败实验 (空指标/无cds) 绝不能产生 git 提交"

    def test_commits_real_metrics(self, monkeypatch):
        calls = self._patch_git(monkeypatch)
        o = make_orchestrator()
        o._git_commit("run_x", 2, "TRAIN",
                      metrics={"cds": 0.42, "precision": 0.9, "recall": 0.8})
        assert calls, "有效指标应正常提交"
        assert calls[0][:2] == ["git", "add"]
        commit_cmd = calls[1]
        assert commit_cmd[:2] == ["git", "commit"]
        assert any("CDS=0.4200" in part for part in commit_cmd)

    def test_explicit_zero_cds_still_commits(self, monkeypatch):
        # cds 键真实存在且为 0.0 — 诚实的坏结果, 允许提交 (与缺失指标区分)
        calls = self._patch_git(monkeypatch)
        o = make_orchestrator()
        o._git_commit("run_x", 3, "TRAIN",
                      metrics={"cds": 0.0, "precision": 0.0, "recall": 0.0})
        assert calls


# ════════════════════════════════════════════════════════════════════
# 修复2b — optuna: 空 metrics 决不以 completed 入库 + 陈旧文件检测
# ════════════════════════════════════════════════════════════════════

def make_runner(state=None, metrics_file="last_metrics.json"):
    r = object.__new__(OptunaRunner)
    r.state = state
    r.study_name = "study_test"
    r.metrics_file = metrics_file
    r._n_with_metrics = 0
    return r


class TestOptunaRecording:
    def test_completed_with_none_metrics_coerced_to_failed(self):
        st = FakeState()
        r = make_runner(st)
        r._record_trial_in_state(1, "rid1", {"LR0": 0.01}, None, "completed", 1.0)
        assert st.inserted[0]["status"] == "failed"
        assert st.inserted[0]["metrics_json"] == {}
        assert st.status_updates[0]["status"] == "failed"

    def test_completed_with_empty_dict_coerced_to_failed(self):
        st = FakeState()
        r = make_runner(st)
        r._record_trial_in_state(2, "rid2", {}, {}, "completed", 1.0)
        assert st.inserted[0]["status"] == "failed"

    def test_completed_with_real_metrics_stays_completed(self):
        st = FakeState()
        r = make_runner(st)
        r._record_trial_in_state(3, "rid3", {}, {"cds": 0.3}, "completed", 1.0)
        assert st.inserted[0]["status"] == "completed"
        assert st.inserted[0]["metrics_json"] == {"cds": 0.3}

    def test_failed_stays_failed(self):
        st = FakeState()
        r = make_runner(st)
        r._record_trial_in_state(4, "rid4", {}, None, "failed", 1.0)
        assert st.inserted[0]["status"] == "failed"


class TestOptunaMetricsFreshness:
    def test_fresh_file_is_read(self, tmp_path):
        mf = tmp_path / "last_metrics.json"
        mf.write_text(json.dumps({"cds": 0.5}), encoding="utf-8")
        r = make_runner(metrics_file=str(mf))
        mtime = os.path.getmtime(str(mf))
        assert r._read_metrics(fresh_after=mtime - 100) == {"cds": 0.5}
        assert r._read_metrics() == {"cds": 0.5}

    def test_stale_file_treated_as_no_metrics(self, tmp_path):
        # 上一轮训练残留的 last_metrics.json — 本轮失败时不能捡来当成果
        mf = tmp_path / "last_metrics.json"
        mf.write_text(json.dumps({"cds": 0.5}), encoding="utf-8")
        r = make_runner(metrics_file=str(mf))
        mtime = os.path.getmtime(str(mf))
        assert r._read_metrics(fresh_after=mtime + 100) is None

    def test_missing_file(self, tmp_path):
        r = make_runner(metrics_file=str(tmp_path / "nope.json"))
        assert r._read_metrics() is None


# ════════════════════════════════════════════════════════════════════
# 修复3 — HPO 断路器: 看「产出非空 metrics 的 trial 数」而非 n_completed
# ════════════════════════════════════════════════════════════════════

class TestHpoBreaker:
    def test_all_completed_but_zero_metrics_is_failure(self):
        # 观察到的原 bug: 失败 trial 计入 completed → 断路器不触发
        result = OptunaSweepResult(n_trials=5, n_completed=5, n_with_metrics=0)
        assert Orchestrator._hpo_sweep_produced_no_metrics(result) is True

    def test_some_metrics_is_success(self):
        result = OptunaSweepResult(n_trials=5, n_completed=5, n_with_metrics=2)
        assert Orchestrator._hpo_sweep_produced_no_metrics(result) is False

    def test_zero_completed_is_failure(self):
        result = OptunaSweepResult(n_trials=5, n_completed=0, n_with_metrics=0)
        assert Orchestrator._hpo_sweep_produced_no_metrics(result) is True

    def test_legacy_result_without_field_falls_back(self):
        legacy_fail = types.SimpleNamespace(n_completed=0)
        legacy_ok = types.SimpleNamespace(n_completed=3)
        assert Orchestrator._hpo_sweep_produced_no_metrics(legacy_fail) is True
        assert Orchestrator._hpo_sweep_produced_no_metrics(legacy_ok) is False

    def test_result_dict_exposes_n_with_metrics(self):
        r = OptunaSweepResult(n_with_metrics=2)
        assert r.as_dict()["n_with_metrics"] == 2
        assert OptunaSweepResult().n_with_metrics == 0


# ════════════════════════════════════════════════════════════════════
# 修复4 — 监督者违规滑动窗口 + 恢复重置
# ════════════════════════════════════════════════════════════════════

def crit_violation(ts):
    return {"timestamp": datetime.fromtimestamp(ts).isoformat(),
            "ts": ts, "level": "CRIT", "msg": "RAM超限"}


def warn_violation(ts):
    return {"timestamp": datetime.fromtimestamp(ts).isoformat(),
            "ts": ts, "level": "WARN", "msg": "RAM偏高"}


class TestSupervisorEmergencyStop:
    def test_trips_on_3_recent_crits(self):
        g = SafetyGuard()
        now = time.time()
        g.violations = [crit_violation(now - 30), crit_violation(now - 20),
                        crit_violation(now - 10)]
        stop, reason = g.should_emergency_stop(now=now)
        assert stop is True
        assert reason

    def test_two_recent_crits_not_enough(self):
        g = SafetyGuard()
        now = time.time()
        g.violations = [crit_violation(now - 30), crit_violation(now - 10)]
        stop, _ = g.should_emergency_stop(now=now)
        assert stop is False

    def test_old_crits_age_out_of_window(self):
        # 修复前: 累计3条 CRIT 后永远建议紧急停止。现在窗口外的不再计入。
        g = SafetyGuard(window_sec=600)
        now = time.time()
        g.violations = [crit_violation(now - 3600), crit_violation(now - 1200),
                        crit_violation(now - 700)]
        stop, _ = g.should_emergency_stop(now=now)
        assert stop is False, "10分钟窗口外的历史违规不应永久触发紧急停止"

    def test_mixed_old_and_fresh(self):
        g = SafetyGuard(window_sec=600)
        now = time.time()
        g.violations = [crit_violation(now - 3600), crit_violation(now - 3000),
                        crit_violation(now - 50), crit_violation(now - 20)]
        assert g.should_emergency_stop(now=now)[0] is False
        g.violations.append(crit_violation(now - 5))
        assert g.should_emergency_stop(now=now)[0] is True

    def test_warns_not_counted(self):
        g = SafetyGuard()
        now = time.time()
        g.violations = [warn_violation(now - 3), warn_violation(now - 2),
                        warn_violation(now - 1)]
        assert g.should_emergency_stop(now=now)[0] is False

    def test_recovery_resets_advice(self):
        g = SafetyGuard()
        now = time.time()
        g.violations = [crit_violation(now - 30), crit_violation(now - 20),
                        crit_violation(now - 10)]
        assert g.should_emergency_stop(now=now)[0] is True

        g.record_recovery(now=now)  # 硬件检查全绿
        assert g.should_emergency_stop(now=now + 1)[0] is False, \
            "恢复后不应继续建议紧急停止"

        # 恢复后又出问题 — 重新从零计数
        g.violations.append(crit_violation(now + 10))
        assert g.should_emergency_stop(now=now + 11)[0] is False
        g.violations.append(crit_violation(now + 12))
        g.violations.append(crit_violation(now + 14))
        assert g.should_emergency_stop(now=now + 15)[0] is True

    def test_legacy_records_without_ts_field(self):
        # 旧格式 (只有 isoformat timestamp) 仍能正确判定
        g = SafetyGuard()
        now = time.time()
        legacy = [{"timestamp": datetime.fromtimestamp(now - 5).isoformat(),
                   "level": "CRIT", "msg": "x"} for _ in range(3)]
        g.violations = legacy
        assert g.should_emergency_stop(now=now)[0] is True

    def test_malformed_timestamp_does_not_crash_or_count(self):
        g = SafetyGuard()
        now = time.time()
        g.violations = [{"timestamp": "garbage", "level": "CRIT", "msg": "x"}
                        for _ in range(5)]
        assert g.should_emergency_stop(now=now)[0] is False

    def test_check_hardware_clean_records_recovery(self, monkeypatch):
        import supervisor_agent as sa
        fake_psutil = types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(used=10, total=100),
            disk_usage=lambda p: types.SimpleNamespace(free=100 * (1024 ** 3)),
        )
        monkeypatch.setattr(sa, "psutil", fake_psutil)
        monkeypatch.setattr(sa, "torch", None)

        g = SafetyGuard()
        now = time.time()
        g.violations = [crit_violation(now - 30), crit_violation(now - 20),
                        crit_violation(now - 10)]
        assert g.should_emergency_stop()[0] is True

        issues = g.check_hardware()
        assert issues == []
        assert g.last_recovery_ts is not None
        assert g.should_emergency_stop()[0] is False, \
            "硬件检查全绿后应自动解除紧急停止建议"

    def test_check_hardware_crit_records_ts_and_caps_list(self, monkeypatch):
        import supervisor_agent as sa
        fake_psutil = types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(used=95, total=100),
            disk_usage=lambda p: types.SimpleNamespace(free=100 * (1024 ** 3)),
        )
        monkeypatch.setattr(sa, "psutil", fake_psutil)
        monkeypatch.setattr(sa, "torch", None)
        monkeypatch.setattr(sa, "write_audit_record", lambda rec: None)

        g = SafetyGuard()
        g.violations = [warn_violation(0)] * sa.VIOLATIONS_KEEP_MAX
        issues = g.check_hardware()
        assert any(level == "CRIT" for level, _ in issues)
        assert "ts" in g.violations[-1]
        assert g.last_recovery_ts is None
        assert len(g.violations) <= sa.VIOLATIONS_KEEP_MAX

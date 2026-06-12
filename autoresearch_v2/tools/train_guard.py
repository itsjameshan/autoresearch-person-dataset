"""train_guard.py — train.py 单例守卫 (singleton training guard)

为什么需要 (2026-06 根因):
  Windows GPU 机上观察到 3 个并发 train.py。链条是:
  OverseerDoctor 经 /api/agent/Orchestrator/restart 重启 Orchestrator
  → 旧 Orchestrator 被 terminate, 但它的 train.py 子进程存活 (Windows 上
    terminate 不级联) → 新 Orchestrator 又 spawn 一个 train.py
  → GPU 抢占导致互相拖慢 → Doctor 再判"卡死"再重启 → 雪崩。

三层防护:
  1. spawn 前单例检查 — psutil 扫 cmdline 含 train.py 的存活进程,
     另加 reports/train.pid 文件双保险 (无 psutil 时的降级路径)。
     发现存活 trainer: 不 spawn, 限时等待; 等不到就放弃本次 (fail-fast)。
  2. Orchestrator 启动时收割孤儿 trainer — 只杀「创建时间早于自身启动」
     且「无存活父进程」的 train.py; 有存活父进程的视为他人管理, 不动。
  3. agent_overseer 停止 agent 时级联终止其子进程树, 不再留孤儿。

本模块刻意零第三方硬依赖: psutil 缺失时退化为 PID 文件方案,
绝不在 Windows 上用 os.kill(pid, 0) 探活 — 那会直接 TerminateProcess。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from typing import Callable, Optional

try:
    import psutil
except ImportError:
    psutil = None

PROJECT_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."
))
PID_FILE = os.path.join(PROJECT_ROOT, "reports", "train.pid")
TRAIN_BASENAME = "train.py"
DEFAULT_POLL_SEC = 5.0
# spawn 前发现并发 trainer 时默认最多等多久 (秒)
SINGLETON_WAIT_SEC = 300.0


def _basename_any(path) -> str:
    """取 basename, 同时认 / 和 \\ 分隔符 (os.path.basename 在 POSIX 上
    不切 Windows 反斜杠路径)。"""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _cmdline_is_trainer(cmdline, train_basename: str = TRAIN_BASENAME) -> bool:
    """cmdline 任一参数的 basename == train.py 即视为训练进程。

    覆盖 [python, -u, C:\\...\\train.py] 与 [python, train.py] 两种形态;
    orchestrator (-m autoresearch_v2.orchestrator) / supervisor 不会误中。
    """
    for arg in (cmdline or []):
        try:
            if _basename_any(arg).lower() == train_basename.lower():
                return True
        except Exception:
            continue
    return False


def _pid_alive_no_psutil(pid) -> bool:
    """无 psutil 时的探活。Windows 走 tasklist — 千万不要 os.kill(pid, 0),
    在 Windows 上那等价于 TerminateProcess(杀掉目标)。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            return str(pid) in (out.stdout or "")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


# ── PID 文件 (reports/train.pid) ───────────────────────────────────────
# run_training 在 Popen 后写入、退出后清除。psutil 缺失时这是唯一线索,
# 有 psutil 时作为双保险 (写入失败不致命, 主防线是 cmdline 扫描)。

def write_pid_file(pid: int, train_script: str = "", path: str = PID_FILE) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "pid": int(pid),
                "train_script": train_script,
                "created": time.time(),
                "created_iso": datetime.now().isoformat(),
            }, f, ensure_ascii=False)
    except Exception:
        pass


def read_pid_file(path: str = PID_FILE) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        int(rec["pid"])
        return rec
    except Exception:
        return None


def clear_pid_file(pid: Optional[int] = None, path: str = PID_FILE) -> None:
    """删除 PID 文件。传入 pid 时只删"记录的就是该 pid"的文件,
    避免误删其他 trainer 刚写入的记录。"""
    try:
        if pid is not None:
            rec = read_pid_file(path)
            if rec is not None and int(rec.get("pid", -1)) != int(pid):
                return
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ── 存活 trainer 探测 ──────────────────────────────────────────────────

def find_live_trainers(exclude_pids=(), train_basename: str = TRAIN_BASENAME,
                       pid_file: str = PID_FILE) -> list[dict]:
    """返回存活 train.py 进程列表: [{pid, cmdline, create_time}, ...]。

    psutil 可用: 全进程扫 cmdline; 不可用: 退化为 PID 文件 + tasklist 探活。
    自身 pid 永远排除。
    """
    exclude = {os.getpid()}
    exclude.update(int(p) for p in exclude_pids)

    if psutil is not None:
        found = []
        try:
            for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
                try:
                    info = proc.info
                    pid = info.get("pid")
                    if pid in exclude:
                        continue
                    if _cmdline_is_trainer(info.get("cmdline"), train_basename):
                        found.append({
                            "pid": pid,
                            "cmdline": list(info.get("cmdline") or []),
                            "create_time": float(info.get("create_time") or 0.0),
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                except Exception:
                    continue
        except Exception:
            return found
        return found

    rec = read_pid_file(pid_file)
    if rec is None:
        return []
    pid = int(rec["pid"])
    if pid in exclude or not _pid_alive_no_psutil(pid):
        return []
    return [{
        "pid": pid,
        "cmdline": [rec.get("train_script", "")],
        "create_time": float(rec.get("created", 0.0)),
    }]


def wait_for_trainers_to_exit(timeout_sec: float,
                              poll_sec: float = DEFAULT_POLL_SEC,
                              exclude_pids=(),
                              train_basename: str = TRAIN_BASENAME,
                              on_wait: Optional[Callable] = None) -> list[dict]:
    """阻塞直到没有存活 trainer 或超时。返回仍存活的列表 ([] = 已清空)。"""
    deadline = time.time() + max(0.0, timeout_sec)
    while True:
        live = find_live_trainers(exclude_pids=exclude_pids,
                                  train_basename=train_basename)
        if not live:
            return []
        now = time.time()
        if now >= deadline:
            return live
        if on_wait is not None:
            try:
                on_wait(live)
            except Exception:
                pass
        time.sleep(max(0.1, min(poll_sec, deadline - now)))


# ── 孤儿收割 / 进程树终止 ──────────────────────────────────────────────

def reap_orphan_trainers(started_before: float, exclude_pids=(),
                         grace_sec: float = 10.0,
                         train_basename: str = TRAIN_BASENAME,
                         pid_file: str = PID_FILE) -> list[dict]:
    """收割孤儿 trainer。只杀同时满足两个条件的 train.py:
      1. create_time 早于 started_before (即早于调用方自身启动 — 绝不是
         调用方刚 spawn 的孩子)
      2. 无存活父进程 (parent 已死 / 被 init 收养) — 有存活父进程说明
         另一个管理进程还持有它, 留给单例等待逻辑处理, 不抢杀。

    需要 psutil (没有它无法确认父进程, 宁可不杀)。返回被收割的进程信息。
    """
    if psutil is None:
        return []
    reaped = []
    for info in find_live_trainers(exclude_pids=exclude_pids,
                                   train_basename=train_basename,
                                   pid_file=pid_file):
        pid = info["pid"]
        try:
            proc = psutil.Process(pid)
            if proc.create_time() >= started_before:
                continue
            parent = None
            try:
                parent = proc.parent()
            except Exception:
                parent = None
            # POSIX 上孤儿会被 init(pid=1) 收养; Windows 上 parent 直接为 None
            has_live_parent = (
                parent is not None
                and parent.is_running()
                and parent.pid != 1
            )
            if has_live_parent:
                continue
            proc.terminate()
            try:
                proc.wait(grace_sec)
            except psutil.TimeoutExpired:
                proc.kill()
            reaped.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            continue

    # 顺手清掉指向已死进程的陈旧 PID 文件
    rec = read_pid_file(pid_file)
    if rec is not None:
        live_pids = {t["pid"] for t in find_live_trainers(
            exclude_pids=exclude_pids, train_basename=train_basename,
            pid_file=pid_file)}
        if int(rec["pid"]) not in live_pids:
            clear_pid_file(path=pid_file)
    return reaped


def terminate_tree(pid: int, include_parent: bool = False,
                   grace_sec: float = 10.0) -> list[int]:
    """终止 pid 的全部子进程 (可选含自身)。agent_overseer.stop() 用它保证
    Orchestrator 被停时其 train.py 子进程一起退出, 不留孤儿。

    需要 psutil; 没有时返回 [] (调用方维持旧行为)。返回已发 terminate 的 pid。
    """
    if psutil is None:
        return []
    try:
        root = psutil.Process(pid)
    except Exception:
        return []
    try:
        procs = root.children(recursive=True)
    except Exception:
        procs = []
    if include_parent:
        procs = procs + [root]
    killed = []
    for p in procs:
        try:
            p.terminate()
            killed.append(p.pid)
        except Exception:
            continue
    if procs:
        try:
            _gone, alive = psutil.wait_procs(procs, timeout=grace_sec)
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass
        except Exception:
            pass
    return killed

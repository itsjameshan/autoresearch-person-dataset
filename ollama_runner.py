"""
ollama_runner.py — Autonomous experiment loop driven by Ollama (local LLM).

Replaces Claude Code / Codex CLI for machines without cloud AI access.
Uses Ollama API (http://localhost:11434). Default model: qwen2.5-coder:14b.

Usage:
  1. Start Ollama (tray app or: ollama serve)
  2. Pull a model: ollama pull qwen2.5-coder:14b
     Optional: set AUTORESEARCH_OLLAMA_MODEL to any name from `ollama list` (e.g. gemma3:4b).
  3. Run: python ollama_runner.py

The runner reads program.md, runs baseline, then autonomously loops:
  modify train.py → commit → train → evaluate → keep/discard → repeat
"""

import os
import re
import sys
import json
import time
import signal
import subprocess
import datetime
import urllib.request
import urllib.error
import threading
from collections import deque

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"
# Override with env if your `ollama list` uses another name (e.g. only gemma3:4b installed).
OLLAMA_MODEL = os.environ.get("AUTORESEARCH_OLLAMA_MODEL", "qwen2.5-coder:14b")
BRANCH = "autoresearch/crowd-win"
MAX_EXPERIMENTS = 20
COOLDOWN_SECONDS = 30        # Less than Mac (RTX 5070 has active cooling)
TRAIN_TIMEOUT = 3600         # 60 minutes max per training run
CDS_KEEP_THRESHOLD = 0.005
TARGET_MET_STREAK_NEEDED = 3
MAX_CONSECUTIVE_DISCARDS = 5

PYTHON = sys.executable      # Use whatever python is running this script
TRAIN_SCRIPT = "train.py"
RUN_LOG = "run.log"
RESULTS_TSV = "results.tsv"
STATUS_MD = "status.md"
SUGGESTIONS_MD = "suggestions.md"


# ══════════════════════════════════════════════════════════════
# 中文进度 / 指标展示
# ══════════════════════════════════════════════════════════════

def _train_line_is_interesting(s: str) -> bool:
    if not s or s.startswith("=") and len(s) > 40:
        return False
    keys = (
        "Epoch", "cds:", "mAP50", "precision", "recall", "f1_optimal",
        "small_obj", "counting_", "inference_ms", "latency_", "peak_memory",
        "---", "train:", "val:", "Speed:", "GPU", "Class", "Images",
        "box_loss", "cls_loss", "EarlyStopping", "patience",
        "Results saved", "Autoresearch", "Error", "Traceback", "WARNING",
        "nan", "CUDA", "epoch", "Optimizer",
    )
    return any(k in s for k in keys)


def format_metrics_report_cn(metrics: dict) -> str:
    """当前轮解析到的指标，中文说明。"""
    if not metrics:
        return "  （未能从 run.log 解析指标，请检查训练是否跑完）"
    lines = [
        "  【本轮核心指标】",
        f"    CDS:              {metrics.get('cds', '?')}",
        f"    mAP50 / mAP50-95: {metrics.get('mAP50', '?')} / {metrics.get('mAP50_95', '?')}",
        f"    Precision / Recall: {metrics.get('precision', '?')} / {metrics.get('recall', '?')}",
        f"    计数 MAE / 小目标召回: {metrics.get('counting_mae', '?')} / {metrics.get('small_obj_recall', '?')}",
        f"    推理耗时(ms) / 延迟分: {metrics.get('inference_ms', '?')} / {metrics.get('latency_score', '?')}",
        f"    门槛: P={metrics.get('precision_gate', '?')}  R={metrics.get('recall_gate', '?')}  延迟={metrics.get('latency_gate', '?')}（上限 {metrics.get('latency_gate_ms', '?')} ms）",
        f"    训练完成 epoch 数: {metrics.get('epochs_completed', '?')}",
    ]
    return "\n".join(lines)


def print_progress_header_cn(
    phase: str,
    session_start: float,
    exp_num: int,
    best_cds: float,
    consecutive_discards: int,
    target_met_streak: int,
    rolling_train_sec: list,
):
    """在终端打印可读的进度与粗略剩余时间（中文）。"""
    elapsed_min = (time.time() - session_start) / 60.0
    # exp_num=0 表示基线；其后循环为 range(start_num, MAX)，此处用粗算上界
    if exp_num == 0:
        remaining_iters = MAX_EXPERIMENTS
    else:
        remaining_iters = max(0, MAX_EXPERIMENTS - exp_num)
    avg_train = (
        sum(rolling_train_sec) / len(rolling_train_sec)
        if rolling_train_sec
        else None
    )
    if avg_train is not None:
        eta_sec = remaining_iters * (avg_train + COOLDOWN_SECONDS)
        eta_str = f"约 {eta_sec / 60:.0f} 分钟（按最近 {len(rolling_train_sec)} 轮训练均值 × 剩余 {remaining_iters} 轮 + 冷却）"
    else:
        eta_str = "尚无耗时样本；完成首轮训练后会估算"

    print("\n" + "─" * 60)
    print(f"【进度】{phase}")
    print(f"  本会话已运行: {elapsed_min:.1f} 分钟")
    print(f"  当前实验编号: #{exp_num}（本分支最多还会尝试约 {remaining_iters} 轮后到达会话上限 {MAX_EXPERIMENTS}）")
    print(f"  历史最佳 CDS: {best_cds:.4f}  |  连续未提升: {consecutive_discards}/{MAX_CONSECUTIVE_DISCARDS}  |  TARGET_MET 连击: {target_met_streak}/{TARGET_MET_STREAK_NEEDED}")
    print(f"  粗略剩余时间: {eta_str}")
    print(f"  （若提前触发「达标连击」或「连续 discard 平台期」会提前结束，实际可能更短）")
    print("─" * 60)


# ══════════════════════════════════════════════════════════════
# OLLAMA API
# ══════════════════════════════════════════════════════════════

def query_ollama(prompt, temperature=0.7, max_tokens=4096):
    """Send a prompt to Ollama and return the response text."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot connect to Ollama at {OLLAMA_URL}")
        print(f"Make sure Ollama is running: ollama serve")
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Ollama query failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# GIT OPERATIONS
# ══════════════════════════════════════════════════════════════

def git(*args):
    """Run a git command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, timeout=30
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_commit(message):
    git("add", TRAIN_SCRIPT)
    return git("commit", "-m", message)


def git_commit_results(message):
    git("add", RESULTS_TSV, STATUS_MD)
    return git("commit", "--amend", "--no-edit")


def git_push():
    return git("push", "origin", BRANCH)


def git_reset_hard():
    return git("reset", "--hard", "HEAD~1")


def git_short_hash():
    _, out, _ = git("rev-parse", "--short", "HEAD")
    return out


# ══════════════════════════════════════════════════════════════
# FILE OPERATIONS
# ══════════════════════════════════════════════════════════════

def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def read_train_config():
    """Extract the EXPERIMENT CONFIG section from train.py."""
    content = read_file(TRAIN_SCRIPT)
    # Find the config section
    match = re.search(
        r"# EXPERIMENT CONFIG.*?# TRAINING — do not modify",
        content, re.DOTALL
    )
    return match.group(0) if match else content[:2000]


def apply_train_changes(new_config_block):
    """Replace the EXPERIMENT CONFIG section in train.py with new values.

    The LLM returns a block of Python variable assignments like:
        MODEL = "yolo12s.pt"
        BATCH = 16
        ...
    We extract valid assignments and update train.py.
    """
    content = read_file(TRAIN_SCRIPT)

    # Parse valid Python assignments from the LLM response
    valid_vars = [
        "MODEL", "IMGSZ", "EPOCHS", "BATCH", "PATIENCE", "DEVICE",
        "LR0", "LRF", "COS_LR", "HSV_H", "HSV_S", "HSV_V",
        "DEGREES", "TRANSLATE", "SCALE", "FLIPUD", "FLIPLR",
        "MOSAIC", "MIXUP", "COPY_PASTE", "ERASING", "CLOSE_MOSAIC",
        "BOX", "CLS", "AMP", "CACHE", "WORKERS", "SINGLE_CLS",
    ]

    for line in new_config_block.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for var in valid_vars:
            pattern = rf'^{var}\s*='
            if re.match(pattern, line):
                # Replace this variable in train.py
                old_pattern = rf'^{var}\s*=.*$'
                content = re.sub(old_pattern, line, content, count=1, flags=re.MULTILINE)
                break

    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.write(content)


# ══════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════

def run_training():
    """Run train.py; stream meaningful lines to terminal; return (success, metrics_dict, duration_sec)."""
    print("\n" + "═" * 60)
    print("【训练+评估】正在执行 train.py")
    print(f"  命令: {PYTHON} {TRAIN_SCRIPT}")
    print("  完整日志: run.log  |  下方实时显示含 Epoch / 指标 / 错误 等关键行")
    print("  另开终端可看滚动日志:  Get-Content .\\run.log -Wait -Tail 12")
    print("═" * 60 + "\n")

    train_start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [PYTHON, TRAIN_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    killer_done = {"v": False}

    def _kill_after_timeout():
        time.sleep(TRAIN_TIMEOUT)
        if not killer_done["v"] and proc.poll() is None:
            proc.kill()
            print(f"\n【超时】训练超过 {TRAIN_TIMEOUT // 60} 分钟，已终止进程。\n")

    threading.Thread(target=_kill_after_timeout, daemon=True).start()

    try:
        with open(RUN_LOG, "w", encoding="utf-8") as log:
            if proc.stdout:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    s = line.rstrip()
                    if _train_line_is_interesting(s):
                        print(f"  │ {s}")
        rc = proc.wait()
    finally:
        killer_done["v"] = True

    success = rc == 0
    duration_sec = time.time() - train_start

    # Parse metrics from run.log
    metrics = {}
    log_content = read_file(RUN_LOG)
    for line in log_content.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("="):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key in ("cds", "mAP50", "mAP50_95", "precision", "recall",
                       "f1_optimal", "small_obj_recall", "counting_acc",
                       "counting_mae", "mean_confidence", "inference_ms",
                       "latency_score", "peak_memory_mb", "epochs_completed",
                       "precision_gate", "recall_gate", "latency_gate",
                       "latency_gate_ms"):
                metrics[key] = val

    if "cds" not in metrics and success:
        if "Error" in log_content or "Traceback" in log_content:
            success = False

    print(f"\n【训练+评估结束】耗时 {duration_sec / 60:.1f} 分钟（{duration_sec:.0f} 秒），退出码 {rc}")
    print(format_metrics_report_cn(metrics if metrics else {}))

    return success, metrics, duration_sec


def parse_cds(metrics):
    """Extract CDS as float from metrics dict."""
    try:
        return float(metrics.get("cds", 0))
    except (ValueError, TypeError):
        return 0.0


def metrics_target_met(metrics):
    """True when precision, recall, and latency gates all PASS (TARGET_MET streak)."""
    if not metrics:
        return False
    if "PASS" not in str(metrics.get("precision_gate", "")):
        return False
    if "PASS" not in str(metrics.get("recall_gate", "")):
        return False
    if "latency_gate" not in metrics:
        return True  # legacy run.log without latency_gate line
    return "PASS" in str(metrics["latency_gate"])


# ══════════════════════════════════════════════════════════════
# RESULTS TRACKING
# ══════════════════════════════════════════════════════════════

def append_result(commit, metrics, status, description):
    """Append a row to results.tsv."""
    cds = metrics.get("cds", "0.0000")
    mAP50 = metrics.get("mAP50", "0.0000")
    mAP50_95 = metrics.get("mAP50_95", "0.0000")
    small_obj = metrics.get("small_obj_recall", "0.0000")
    precision = metrics.get("precision", "0.0000")
    recall = metrics.get("recall", "0.0000")
    mae = metrics.get("counting_mae", "0.0")
    inf_ms = metrics.get("inference_ms", "0.0")
    lat_sc = metrics.get("latency_score", "0.0000")
    mem = metrics.get("peak_memory_mb", "0.0")
    epochs = metrics.get("epochs_completed", "0")

    try:
        mem_gb = f"{float(mem) / 1024:.1f}"
    except (ValueError, TypeError):
        mem_gb = "0.0"

    row = f"{commit}\t{cds}\t{mAP50}\t{mAP50_95}\t{small_obj}\t{precision}\t{recall}\t{mae}\t{inf_ms}\t{lat_sc}\t{mem_gb}\t{epochs}\t{status}\t{description}\n"

    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(row)


def get_results_history():
    """Read results.tsv and return as string."""
    return read_file(RESULTS_TSV)


def update_status(experiment_num, best_cds, best_commit, best_desc,
                  metrics, target_met_streak, consecutive_discards,
                  last_experiments, what_worked, next_experiment):
    """Update status.md dashboard."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    precision = metrics.get("precision", "?")
    recall = metrics.get("recall", "?")
    inference_ms = metrics.get("inference_ms", "?")
    latency_gate = metrics.get("latency_gate", "?")
    latency_cap = metrics.get("latency_gate_ms", "?")

    last_exp_table = ""
    for exp in last_experiments[-3:]:
        last_exp_table += f"| {exp['num']} | {exp['cds']} | {exp['status']} | {exp['desc']} |\n"

    content = f"""# Autoresearch Status

## Session Info
- Updated: {now}
- Branch: {BRANCH}
- Total experiments: {experiment_num}
- Platform: Windows / RTX 5070 12GB / 64GB RAM / Ollama {OLLAMA_MODEL}

## Current Best
- CDS: {best_cds} (commit: {best_commit})
- Description: {best_desc}

## Quality Gate Progress
- Precision: {precision}/0.90 target
- Recall: {recall}/0.85 target
- Latency: {inference_ms} ms mean (tile) — {latency_gate} (cap {latency_cap} ms; override env AUTORESEARCH_LATENCY_GATE_MS)
- TARGET_MET streak: {target_met_streak}/{TARGET_MET_STREAK_NEEDED} needed to stop (P/R + latency all PASS)

## Stop Condition Status
- Consecutive discards: {consecutive_discards}/{MAX_CONSECUTIVE_DISCARDS}
- Session experiments: {experiment_num}/{MAX_EXPERIMENTS}
- Hardware: RTX 5070 (active cooling, no thermal concern)

## Last 3 Experiments
| # | CDS | Status | Description |
|---|-----|--------|-------------|
{last_exp_table}
## What's Working
{what_worked}

## Next Experiment
{next_experiment}
"""
    with open(STATUS_MD, "w", encoding="utf-8") as f:
        f.write(content)


# ══════════════════════════════════════════════════════════════
# LLM-DRIVEN EXPERIMENT GENERATION
# ══════════════════════════════════════════════════════════════

def generate_experiment(experiment_num, current_config, results_history, suggestions):
    """Ask Ollama to propose the next experiment modification."""

    prompt = f"""You are an autonomous ML researcher optimizing a YOLOv12 dense crowd detection model.
Your goal: maximize CDS (Crowd Detection Score). CDS includes ~10% weight on latency (faster inference -> higher score), plus mAP/F1/counting/small-object terms. TARGET_MET requires precision>=0.90, recall>=0.85, AND mean inference_ms <= latency cap (default in evaluate.py; set AUTORESEARCH_LATENCY_GATE_MS for client hardware).

Hardware: Windows, RTX 5070 12GB VRAM, 64GB RAM, i5-14600KF.
Constraints: batch<=16 at imgsz=1280 (12GB VRAM), cache="ram" is fine (64GB).

Current train.py config:
```python
{current_config}
```

Experiment history (results.tsv):
```
{results_history}
```

Suggestions from advisor:
```
{suggestions}
```

This is experiment #{experiment_num}. Based on the history, propose ONE specific change to train.py.

Rules:
- Only change variables in the EXPERIMENT CONFIG section
- Available models: yolov8s.pt, yolov8n.pt, yolo12n.pt, yolo12s.pt, yolo12l.pt
- Do NOT change DEVICE, DATA_YAML, or anything below "TRAINING — do not modify"
- Keep changes focused — one hypothesis per experiment
- If recent experiments failed, try something different

Respond with EXACTLY:
1. One line: DESCRIPTION: <what you're trying and why>
2. Then ONLY the Python variable assignments that change, e.g.:
MODEL = "yolo12s.pt"
BATCH = 12

Do not include unchanged variables. Do not include explanations after the assignments.
"""

    response = query_ollama(prompt, temperature=0.7)
    if not response:
        return None, None

    # Parse response
    lines = response.strip().split("\n")
    description = ""
    config_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("DESCRIPTION:"):
            description = line.replace("DESCRIPTION:", "").strip()
        elif re.match(r'^[A-Z_]+\s*=', line):
            # Remove markdown code fence artifacts
            clean = line.replace("`", "").strip()
            config_lines.append(clean)

    if not description:
        description = f"ollama experiment #{experiment_num}"

    config_block = "\n".join(config_lines)
    return description, config_block


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    session_start = time.time()
    rolling_train_sec = deque(maxlen=5)

    print("=" * 60)
    print("AUTORESEARCH — Ollama 自主实验循环（中文进度说明）")
    print(f"  模型: {OLLAMA_MODEL}")
    print(f"  分支: {BRANCH}")
    print(f"  会话内最多实验轮次上限: {MAX_EXPERIMENTS}（含基线占用的一轮逻辑，见下）")
    print(f"  训练超时: {TRAIN_TIMEOUT // 60} 分钟  |  轮间冷却: {COOLDOWN_SECONDS} 秒")
    print("=" * 60)

    # Check Ollama connectivity
    test = query_ollama("Say OK", max_tokens=10)
    if test is None:
        print("\n【致命错误】无法连接 Ollama，请确认托盘或 ollama serve 已运行。")
        sys.exit(1)
    print(f"【就绪】Ollama 已连接，使用模型: {OLLAMA_MODEL}")

    # Check git branch
    rc, branch, _ = git("branch", "--show-current")
    if branch != BRANCH:
        print(f"Switching to branch {BRANCH}...")
        git("checkout", BRANCH)

    # State
    best_cds = 0.0
    best_commit = "N/A"
    best_desc = "N/A"
    target_met_streak = 0
    consecutive_discards = 0
    last_experiments = []
    what_worked = "(pending first experiments)"

    # Read existing results to resume state
    existing = read_file(RESULTS_TSV)
    existing_lines = [l for l in existing.strip().split("\n")[1:] if l.strip()]
    start_num = len(existing_lines)

    for line in existing_lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                cds = float(parts[1])
                if cds > best_cds:
                    best_cds = cds
                    best_commit = parts[0]
                    best_desc = parts[-1] if len(parts) >= 14 else "previous"
            except ValueError:
                pass

    # ── BASELINE (if no results yet) ──
    if start_num == 0:
        print_progress_header_cn(
            "基线：不修改 train.py，直接训练+评估当前配置",
            session_start, 0, best_cds, 0, 0, list(rolling_train_sec),
        )
        print("【步骤】提交基线 git commit 并启动 train.py …\n")
        git_commit("experiment: baseline — " + read_train_config()[:80].replace("\n", " "))
        success, metrics, dur = run_training()
        rolling_train_sec.append(dur)
        commit = git_short_hash()

        if success and "cds" in metrics:
            cds = parse_cds(metrics)
            best_cds = cds
            best_commit = commit
            best_desc = "baseline"
            target_met = metrics_target_met(metrics)

            append_result(commit, metrics, "keep", "baseline" + (" [TARGET_MET]" if target_met else ""))
            git("add", RESULTS_TSV, STATUS_MD)
            git("commit", "--amend", "--no-edit")
            git_push()
            print(f"\n【基线完成】CDS = {cds:.4f}" + ("  ✅ 已满足 TARGET_MET 门槛" if target_met else ""))
            last_experiments.append({"num": 0, "cds": cds, "status": "keep", "desc": "baseline"})
        else:
            print("\n【基线失败】请查看上方指标与 run.log 尾部。")
            tail = read_file(RUN_LOG)[-500:]
            print(tail)
            append_result(commit, metrics, "crash", "baseline crash")
            git_reset_hard()
            # Still continue — maybe the config needs adjustment

        start_num = 1

    # ── EXPERIMENT LOOP ──
    for exp_num in range(start_num, MAX_EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"实验轮次 #{exp_num} / 编号小于 {MAX_EXPERIMENTS} 即在本会话范围内")
        print(f"{'='*60}")

        print_progress_header_cn(
            f"第 {exp_num} 轮：Ollama 假设 → 训练 → 评估",
            session_start, exp_num, best_cds, consecutive_discards, target_met_streak,
            list(rolling_train_sec),
        )

        # Check stop conditions
        if target_met_streak >= TARGET_MET_STREAK_NEEDED:
            print(f"\n【停止条件】已连续 {TARGET_MET_STREAK_NEEDED} 次 KEEP 且达标 — 结束会话")
            git_push()
            break

        if consecutive_discards >= MAX_CONSECUTIVE_DISCARDS:
            print(f"\n【停止条件】已连续 {MAX_CONSECUTIVE_DISCARDS} 次未提升 — 判定平台期，结束会话")
            # Write analysis
            analysis = query_ollama(
                f"Analyze these experiment results and suggest new directions:\n{get_results_history()}",
                max_tokens=1000
            )
            if analysis:
                with open(SUGGESTIONS_MD, "w", encoding="utf-8") as f:
                    f.write(f"# Plateau Analysis (after {exp_num} experiments)\n\n{analysis}\n")
            git_push()
            break

        # Generate next experiment via Ollama
        print("【步骤】正在调用 Ollama 根据 program.md / results.tsv 生成 train.py 修改建议 …")
        current_config = read_train_config()
        results_history = get_results_history()
        suggestions = read_file(SUGGESTIONS_MD)

        description, config_block = generate_experiment(
            exp_num, current_config, results_history, suggestions
        )

        if not config_block:
            print("【提示】Ollama 未返回可解析的配置行，提高温度重试一次 …")
            description, config_block = generate_experiment(
                exp_num, current_config, results_history, suggestions
            )
            if not config_block:
                print("【跳过】仍无有效修改，本回合不计入训练，连续 discard +1")
                consecutive_discards += 1
                continue

        # Apply changes
        print(f"\n【假设】{description}")
        print(f"【将写入 train.py 的赋值】\n{config_block}\n")
        apply_train_changes(config_block)

        # Git commit
        git_commit(f"experiment: {description}")

        # Run training
        success, metrics, dur = run_training()
        rolling_train_sec.append(dur)
        commit = git_short_hash()

        if not success or "cds" not in metrics:
            # CRASH
            print("【崩溃】训练或评估失败，正在展示 run.log 片段 …")
            tail = read_file(RUN_LOG)[-500:]
            print(tail[-300:])
            append_result(commit, {}, "crash", f"crash: {description}")
            git_reset_hard()
            # Commit just the results.tsv
            git("add", RESULTS_TSV)
            git("commit", "-m", f"log: crash — {description}")
            git_push()
            consecutive_discards += 1
            last_experiments.append({"num": exp_num, "cds": "CRASH", "status": "crash", "desc": description})
        else:
            cds = parse_cds(metrics)
            improvement = cds - best_cds
            target_met = metrics_target_met(metrics)

            print(f"\n【决策输入】本轮 CDS={cds:.4f}  |  历史最佳={best_cds:.4f}  |  提升={improvement:+.4f}  |  保留阈值>{CDS_KEEP_THRESHOLD}")

            if improvement > CDS_KEEP_THRESHOLD:
                # KEEP
                status_str = "keep" + (" [TARGET_MET]" if target_met else "")
                append_result(commit, metrics, status_str, description)
                best_cds = cds
                best_commit = commit
                best_desc = description
                consecutive_discards = 0

                if target_met:
                    target_met_streak += 1
                else:
                    target_met_streak = 0

                git("add", RESULTS_TSV, STATUS_MD)
                git("commit", "--amend", "--no-edit")
                git_push()
                print(f"【结果】✅ KEEP — CDS 提升 {improvement:+.4f}，已 amend 提交并 push")
                if target_met:
                    print(f"  本轮同时满足 TARGET_MET（连击 {target_met_streak}/{TARGET_MET_STREAK_NEEDED}）")

                last_experiments.append({"num": exp_num, "cds": cds, "status": "keep", "desc": description})
                what_worked = "\n".join(
                    f"- {e['desc']} (CDS: {e['cds']})"
                    for e in last_experiments if e["status"] == "keep"
                )[-500:]
            else:
                # DISCARD
                append_result(commit, metrics, "discard", description)
                git_reset_hard()
                # Commit results.tsv separately
                git("add", RESULTS_TSV)
                git("commit", "-m", f"log: discard — {description}")
                git_push()
                consecutive_discards += 1
                target_met_streak = 0
                print(f"【结果】❌ DISCARD — CDS 变化 {improvement:+.4f} 未超过阈值，已 git reset 丢弃本轮 train.py")

                last_experiments.append({"num": exp_num, "cds": cds, "status": "discard", "desc": description})

        # Update status dashboard
        update_status(
            exp_num + 1, best_cds, best_commit, best_desc,
            metrics, target_met_streak, consecutive_discards,
            last_experiments, what_worked,
            "generating next hypothesis..."
        )

        # Cooldown
        total_min = (time.time() - session_start) / 60.0
        print(f"\n【冷却】等待 {COOLDOWN_SECONDS} 秒后开始下一轮…（本会话已累计 {total_min:.1f} 分钟）")
        time.sleep(COOLDOWN_SECONDS)

    # Final summary
    total_session_min = (time.time() - session_start) / 60.0
    print("\n" + "=" * 60)
    print("【会话结束】AUTORESEARCH")
    print(f"  本终端记录到的实验条目数: {len(last_experiments)}")
    print(f"  最佳 CDS: {best_cds:.4f}  （commit {best_commit}）")
    print(f"  对应描述: {best_desc}")
    print(f"  本会话总耗时: {total_session_min:.1f} 分钟")
    print("  详情仍见: results.tsv 、 status.md 、 run.log")
    print("=" * 60)

    # Final push
    git("add", RESULTS_TSV, STATUS_MD)
    git("commit", "-m", "autoresearch: session complete")
    git_push()


if __name__ == "__main__":
    main()

"""
ollama_runner.py — Autonomous experiment loop driven by Ollama (local LLM).

Replaces Claude Code / Codex CLI for machines without cloud AI access.
Uses Ollama API (http://localhost:11434). Default model: qwen2.5-coder:14b（代码生成
能力远强于通用模型，推荐用于 autoresearch）。

Usage:
  1. Start Ollama in CPU mode (keep GPU free for YOLO training):
       $env:OLLAMA_GPU_LAYERS = 0; ollama serve
  2. Pull model: ollama pull qwen2.5-coder:14b
  3. Optional: $env:AUTORESEARCH_OLLAMA_MODEL = "<exact name from ollama list>"
  4. Preflight check: python ollama_runner.py --preflight
  5. Run: python ollama_runner.py
     全程无交互，直到：达标连击 / 连续 discard 平台期 / 满 MAX_EXPERIMENTS / 你 Ctrl+C。
     若 Git push 弹窗打断：先配置凭据，或 PowerShell 临时跳过远程同步：
       $env:AUTORESEARCH_SKIP_PUSH="1"; python ollama_runner.py
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
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
# main() 会用本机 /api/tags 校正为「ollama list」里的准确名称，减少 404。
OLLAMA_MODEL = os.environ.get("AUTORESEARCH_OLLAMA_MODEL", "qwen2.5-coder:14b").strip() or "qwen2.5-coder:14b"
BRANCH = "autoresearch/crowd-win"
MAX_EXPERIMENTS = 20
COOLDOWN_SECONDS = 30        # Less than Mac (RTX 5070 has active cooling)
TRAIN_TIMEOUT = 14400  # 4 小时，根据你实际训练时间调整
CDS_KEEP_THRESHOLD = 0.005
TARGET_MET_STREAK_NEEDED = 3
MAX_CONSECUTIVE_DISCARDS = 5

PYTHON = sys.executable      # Use whatever python is running this script
TRAIN_SCRIPT = "train.py"
RUN_LOG = "run.log"
METRICS_JSON = "last_metrics.json"
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

def _ollama_cli_hint():
    exe = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    if os.path.isfile(exe):
        return f'& "{exe}" list'
    return "ollama list  （若未加入 PATH，请用开始菜单安装目录下的 ollama.exe）"


def resolve_ollama_model_name(wanted: str) -> str:
    """
    将 wanted 映射为本机已安装列表中的准确名称（避免 qwen2.5-coder:14b vs :latest 等导致 HTTP 404）。
    """
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL)
        with urllib.request.urlopen(req, timeout=15) as resp:
            names = [m.get("name", "") for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
    except Exception as e:
        print(f"[WARN] 无法读取 Ollama 模型列表 ({e})，将按原样使用: {wanted!r}")
        return wanted

    if not names:
        print("[FATAL] Ollama 中没有任何模型。请执行 pull 后再运行。")
        print(f"  {_ollama_cli_hint()}")
        sys.exit(1)

    if wanted in names:
        return wanted

    base = wanted.split(":")[0] if ":" in wanted else wanted
    for n in names:
        if n == wanted or n.startswith(wanted + ":"):
            print(f"[INFO] 模型名已对齐: {wanted!r} → {n!r}（以本机 ollama list 为准）")
            return n

    for n in names:
        if n.startswith(base + ":") or n == base:
            print(f"[INFO] 模型名已对齐: {wanted!r} → {n!r}")
            return n

    # 请求的模型尚未 pull 完：优先用本机已有的 gemma3，便于先跑 autoresearch
    for n in names:
        if n.startswith("gemma3"):
            print(f"[WARN] 未找到 {wanted!r}（可能仍在下载），暂用本机已有: {n!r}")
            return n
    for n in names:
        if "qwen" in n.lower():
            print(f"[WARN] 未找到 {wanted!r}，暂用本机已有: {n!r}")
            return n

    print(f"[FATAL] 未找到与 {wanted!r} 匹配的模型，且本机无 gemma/qwen 可降级。已有列表:")
    for n in names:
        print(f"    {n}")
    print(f"  请设置: $env:AUTORESEARCH_OLLAMA_MODEL = \"<上列之一>\"")
    print(f"  或执行: {_ollama_cli_hint()}")
    sys.exit(1)


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
    except urllib.error.HTTPError as e:
        print(f"ERROR: Ollama HTTP {e.code} at {OLLAMA_URL}")
        if e.code == 404:
            print(f"  多为模型名不对：当前请求 model={OLLAMA_MODEL!r}")
            print(f"  请运行 {_ollama_cli_hint()} 把名称设成与 NAME 列完全一致（或删掉环境变量让脚本自动对齐）。")
        try:
            b = e.read().decode("utf-8", errors="replace")[:400]
            if b.strip():
                print(f"  Body: {b.strip()}")
        except Exception:
            pass
        return None
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach Ollama at {OLLAMA_URL}")
        print("  确认托盘里 Ollama 已启动；勿重复 ollama serve（端口占用说明服务已在跑）。")
        print(f"  {e}")
        return None
    except Exception as e:
        print(f"ERROR: Ollama query failed: {e}")
        return None


def unload_ollama_model():
    """Ask Ollama to unload the model from VRAM before training."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "keep_alive": 0,
        "prompt": "",
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        print("  [INFO] Ollama model unloaded from VRAM")
    except Exception:
        pass  # Best effort


# ══════════════════════════════════════════════════════════════
# VRAM SAFETY — validate LLM-proposed config before training
# ══════════════════════════════════════════════════════════════

# Estimated peak VRAM (GB) for model+imgsz combos at batch=1, scaled by batch size
_VRAM_ESTIMATES = {
    # (model_keyword, imgsz) → approximate VRAM in GB at batch=8
    ("yolov8n", 640): 2.0,
    ("yolov8s", 640): 3.0,
    ("yolov8s", 1280): 8.0,
    ("yolo12n", 640): 2.5,
    ("yolo12s", 640): 5.0,
    ("yolo12s", 1280): 25.0,  # OVERFLOWS 12GB!
    ("yolo12l", 640): 10.0,
    ("yolo12l", 1280): 40.0,  # WAY over 12GB
}

GPU_VRAM_LIMIT_GB = 12.0


def estimate_vram(model_str, imgsz, batch):
    """Rough VRAM estimate. Returns (estimated_gb, safe)."""
    model_lower = model_str.lower()
    for key_model in ("yolo12l", "yolo12s", "yolo12n", "yolov8s", "yolov8n"):
        if key_model in model_lower:
            base = _VRAM_ESTIMATES.get((key_model, imgsz))
            if base is None:
                # Interpolate: VRAM roughly scales with imgsz^2
                base640 = _VRAM_ESTIMATES.get((key_model, 640), 4.0)
                base = base640 * (imgsz / 640) ** 2
            # Scale by batch (rough: VRAM ~ batch * per-image activations)
            estimated = base * (batch / 8)
            return estimated, estimated < GPU_VRAM_LIMIT_GB
    return None, True  # Unknown model, let it try


def validate_train_config(train_content):
    """Check if the current train.py config will fit in VRAM. Returns (ok, message)."""
    model_match = re.search(r'^MODEL\s*=\s*["\'](.+?)["\']', train_content, re.MULTILINE)
    imgsz_match = re.search(r'^IMGSZ\s*=\s*(\d+)', train_content, re.MULTILINE)
    batch_match = re.search(r'^BATCH\s*=\s*(\d+)', train_content, re.MULTILINE)
    copy_paste_match = re.search(r'^COPY_PASTE\s*=\s*([\d.]+)', train_content, re.MULTILINE)
    mixup_match = re.search(r'^MIXUP\s*=\s*([\d.]+)', train_content, re.MULTILINE)

    if not (model_match and imgsz_match and batch_match):
        return True, "Could not parse config"

    model = model_match.group(1)
    imgsz = int(imgsz_match.group(1))
    batch = int(batch_match.group(1))
    copy_paste = float(copy_paste_match.group(1)) if copy_paste_match else 0.0
    mixup = float(mixup_match.group(1)) if mixup_match else 0.0

    est, safe = estimate_vram(model, imgsz, batch)

    # Heavy augmentation (copy_paste/mixup > 0.2) roughly 1.5x VRAM
    if copy_paste > 0.2 or mixup > 0.2:
        if est:
            est *= 1.5
            safe = est < GPU_VRAM_LIMIT_GB

    if est and not safe:
        return False, (
            f"VRAM UNSAFE: {model} imgsz={imgsz} batch={batch} "
            f"(copy_paste={copy_paste}, mixup={mixup}) → ~{est:.1f}GB, "
            f"exceeds {GPU_VRAM_LIMIT_GB}GB limit"
        )
    return True, f"VRAM OK: ~{est:.1f}GB" if est else "VRAM: unknown model, proceeding"


def fix_unsafe_config(train_content):
    """If config is VRAM-unsafe, downgrade to safe defaults and return fixed content."""
    ok, msg = validate_train_config(train_content)
    if ok:
        return train_content, msg

    print(f"  [VRAM GUARD] {msg}")
    print(f"  [VRAM GUARD] Auto-fixing to safe defaults...")

    # Apply safe defaults
    fixes = {
        "MODEL": '"person_dataset/yolov8s.pt"',
        "IMGSZ": "640",
        "BATCH": "16",
        "COPY_PASTE": "0.1",
        "MIXUP": "0.1",
    }
    for var, val in fixes.items():
        train_content = re.sub(
            rf'^{var}\s*=.*$', f'{var} = {val}',
            train_content, count=1, flags=re.MULTILINE
        )

    ok2, msg2 = validate_train_config(train_content)
    print(f"  [VRAM GUARD] After fix: {msg2}")
    return train_content, msg2


# ══════════════════════════════════════════════════════════════
# GIT OPERATIONS
# ══════════════════════════════════════════════════════════════

def git(*args):
    """Run a git command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace'
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_commit(message):
    git("add", TRAIN_SCRIPT)
    return git("commit", "-m", message)


def git_commit_results(message):
    git("add", RESULTS_TSV, STATUS_MD)
    return git("commit", "--amend", "--no-edit")


def git_push():
    """远程 push；设 AUTORESEARCH_SKIP_PUSH=1 可跳过（避免无人值守时弹 Git 凭据窗）。"""
    if os.environ.get("AUTORESEARCH_SKIP_PUSH", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        print("  （已跳过 git push：AUTORESEARCH_SKIP_PUSH 已设置，结束后再手动 push）")
        return 0, "", ""
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
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


_METRIC_KEYS = (
    "cds", "mAP50", "mAP50_95", "precision", "recall",
    "f1_optimal", "small_obj_recall", "counting_acc",
    "counting_mae", "mean_confidence", "inference_ms",
    "latency_score", "peak_memory_mb", "epochs_completed",
    "precision_gate", "recall_gate", "latency_gate",
    "latency_gate_ms",
)


def _load_metrics_json(path=METRICS_JSON):
    """Load metrics dict from JSON written by evaluate.py.

    Returns dict with all values stringified (for parity with the legacy
    log-parsing path), or None if the file is missing/corrupt.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  （读取 {path} 失败: {e}，回退到 run.log 解析）")
        return None
    if not isinstance(raw, dict):
        return None
    return {k: str(raw[k]) for k in _METRIC_KEYS if k in raw}


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
        MODEL = "person_dataset/yolo12s.pt"
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

    # Drop any stale metrics JSON from a previous run so we never read old data.
    if os.path.exists(METRICS_JSON):
        try:
            os.remove(METRICS_JSON)
        except OSError as e:
            print(f"  （无法删除旧 {METRICS_JSON}: {e}）")

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
        encoding='utf-8',
        errors='replace'
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

    # Prefer the metrics JSON written by evaluate.py (robust to log format
    # changes). Fall back to regex parsing of run.log for legacy/partial runs.
    metrics = _load_metrics_json() or {}
    log_content = read_file(RUN_LOG)
    if not metrics:
        for line in log_content.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("="):
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if key in _METRIC_KEYS:
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
    prompt = f"""You are an autonomous ML researcher optimizing a YOLO dense crowd detection model.
Your goal: maximize CDS (Crowd Detection Score).

Hardware: Windows, RTX 5070 12GB VRAM, 64GB RAM, i5-14600KF.

=== CRITICAL VRAM CONSTRAINTS (MUST OBEY) ===
- GPU has ONLY 12GB VRAM. Exceeding this causes 10-15x slowdown (unified memory fallback).
- VRAM budget table (approximate peak at batch=8):
    yolov8n + 640  → ~2GB  ✓
    yolov8s + 640  → ~3GB  ✓
    yolov8s + 1280 → ~8GB  ✓ (safe with batch<=8)
    yolo12n + 640  → ~2.5GB ✓
    yolo12s + 640  → ~5GB  ✓
    yolo12s + 1280 → ~25GB ✗ WILL OVERFLOW, DO NOT USE
    yolo12l + ANY  → TOO LARGE, DO NOT USE
- NEVER set COPY_PASTE > 0.2 or MIXUP > 0.2 — these nearly double VRAM usage.
- Safe combos: yolov8s+1280+batch8, yolov8s+640+batch16, yolo12s+640+batch8
- Model paths must use "person_dataset/" prefix, e.g. "person_dataset/yolov8s.pt"
- Available models: person_dataset/yolov8n.pt, person_dataset/yolov8s.pt, person_dataset/yolo12n.pt, person_dataset/yolo12s.pt
- Do NOT change DEVICE, DATA_YAML, CACHE, WORKERS, or anything below "TRAINING — do not modify"

Current train.py config:
{current_config}

Experiment history:
{results_history}

Suggestions:
{suggestions}

This is experiment #{experiment_num}.
Propose ONE specific change to train.py. Focus on what will improve CDS most.

Respond EXACTLY in this format (no other text):
DESCRIPTION: <what you're trying and why>
VAR = value
VAR2 = value

Only include variables that CHANGE. Do not repeat unchanged variables.
"""

    response = query_ollama(prompt, temperature=0.7)
    if not response:
        return None, None

    lines = response.strip().split("\n")
    description = ""
    config_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("DESCRIPTION:"):
            description = line.replace("DESCRIPTION:", "").strip()
        elif "=" in line and line.split("=")[0].strip().isupper():
            clean = line.replace("`", "").strip()
            config_lines.append(clean)

    if not description:
        description = f"experiment {experiment_num}"

    config_block = "\n".join(config_lines)
    return description, config_block

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    global OLLAMA_MODEL  # ⬅️ 必须放在最前面
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "Ollama Runner - 自主实验循环" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"  分支: {BRANCH}")
    print(f"  模型: {OLLAMA_MODEL} (环境变量 AUTORESEARCH_OLLAMA_MODEL 可覆盖)")
    print(f"  实验上限: {MAX_EXPERIMENTS} 轮 | 达标连击要求: {TARGET_MET_STREAK_NEEDED} | 连续未提升上限: {MAX_CONSECUTIVE_DISCARDS}")
    print()

    # 解析最终使用的模型名（与 Ollama list 对齐）
    OLLAMA_MODEL = resolve_ollama_model_name(OLLAMA_MODEL)
    print(f"[INFO] 实际请求 Ollama 模型名: {OLLAMA_MODEL!r}")

    # 检查必要文件
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"[FATAL] 未找到训练脚本 {TRAIN_SCRIPT}，请在项目目录下运行。")
        sys.exit(1)

    # Git 准备：切换到目标分支（若不存在则创建）
    rc, out, err = git("checkout", BRANCH)
    if rc != 0:
        print(f"[INFO] 分支 {BRANCH} 不存在，尝试创建...")
        rc2, _, _ = git("checkout", "-b", BRANCH)
        if rc2 != 0:
            print(f"[WARN] Git 分支操作失败，但继续执行（可能已在该分支）。")

    # 初始化结果文件（如不存在）
    if not os.path.exists(RESULTS_TSV):
        header = "commit\tcds\tmAP50\tmAP50_95\tsmall_obj_recall\tprecision\trecall\tcounting_mae\tinference_ms\tlatency_score\tpeak_memory_gb\tepochs\tstatus\tdescription\n"
        with open(RESULTS_TSV, "w", encoding="utf-8") as f:
            f.write(header)

    # 读取 suggestions.md（如有）
    suggestions = read_file(SUGGESTIONS_MD) or "（无预设建议，请基于当前结果探索）"

    # 状态变量
    session_start = time.time()
    best_cds = -1.0
    best_commit = ""
    best_desc = ""
    best_metrics = {}
    consecutive_discards = 0
    target_met_streak = 0
    experiment_num = 0
    last_experiments = []          # 每个元素为 dict: num, cds, status, desc
    rolling_train_sec = deque(maxlen=5)  # 用于估算剩余时间

    # 信号处理（Ctrl+C 优雅退出）
    stop_requested = {"value": False}
    def signal_handler(sig, frame):
        print("\n[用户中断] 正在安全退出...")
        stop_requested["value"] = True
    signal.signal(signal.SIGINT, signal_handler)

    # ─────────────────────────────────────────────────────────────
    # 基线实验（实验 #0）
    # ─────────────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("【阶段 1/2】运行基线训练（当前 train.py 配置）")
    print("█" * 60)

    # 基线前验证 VRAM 安全
    baseline_content = read_file(TRAIN_SCRIPT)
    fixed_content, vram_msg = fix_unsafe_config(baseline_content)
    if fixed_content != baseline_content:
        with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
            f.write(fixed_content)
        print(f"  [VRAM GUARD] 基线配置已自动修正")
    else:
        print(f"  [VRAM CHECK] {vram_msg}")

    # 释放 Ollama 显存再训练
    unload_ollama_model()

    success, metrics, duration = run_training()
    rolling_train_sec.append(duration)

    if not success:
        print("[FATAL] 基线训练失败，请检查 train.py 与环境后重试。")
        sys.exit(1)

    cds = parse_cds(metrics)
    commit = git_short_hash()
    description = "baseline"
    status = "baseline"
    append_result(commit, metrics, status, description)
    last_experiments.append({"num": 0, "cds": f"{cds:.4f}", "status": status, "desc": description})

    # 基线作为当前最佳
    best_cds = cds
    best_commit = commit
    best_desc = description
    best_metrics = metrics

    # Git 提交基线（如未提交）
    rc, _, _ = git("diff", "--quiet", TRAIN_SCRIPT)
    if rc != 0:
        git_commit(f"baseline: CDS={cds:.4f}")
        git_commit_results(f"baseline results")

    # 判断是否直接达标
    if metrics_target_met(metrics):
        target_met_streak = 1
        print(f"  基线即满足质量门 (P/R/latency 全部 PASS)，TARGET_MET 连击 = {target_met_streak}")
    else:
        target_met_streak = 0

    # 打印进度头
    print_progress_header_cn(
        phase="基线完成，即将开始自主实验循环",
        session_start=session_start,
        exp_num=0,
        best_cds=best_cds,
        consecutive_discards=consecutive_discards,
        target_met_streak=target_met_streak,
        rolling_train_sec=list(rolling_train_sec),
    )

    # ─────────────────────────────────────────────────────────────
    # 自主实验循环
    # ─────────────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("【阶段 2/2】进入自主实验循环")
    print("█" * 60)

    while experiment_num < MAX_EXPERIMENTS:
        if stop_requested["value"]:
            print("[用户中断] 退出循环。")
            break

        experiment_num += 1
        print(f"\n{'#' * 60}")
        print(f"【实验 #{experiment_num}】生成实验修改...")
        print(f"{'#' * 60}")

        # 读取当前配置
        current_config = read_train_config()
        results_history = get_results_history()
        if not results_history:
            results_history = "（暂无历史记录）"

        # 调用 LLM 生成实验
        description, config_block = generate_experiment(
            experiment_num, current_config, results_history, suggestions
        )
        if description is None or config_block is None:
            print("[ERROR] LLM 返回无效响应，跳过本轮，等待冷却后重试。")
            time.sleep(COOLDOWN_SECONDS)
            continue

        print(f"  LLM 建议: {description}")
        print("  修改配置:")
        for line in config_block.split("\n"):
            if line.strip():
                print(f"    {line.strip()}")

        # 备份当前 train.py（以防修改失败）
        backup_content = read_file(TRAIN_SCRIPT)

        # 应用修改
        try:
            apply_train_changes(config_block)
        except Exception as e:
            print(f"[ERROR] 应用修改时出错: {e}，恢复 train.py 并跳过本轮。")
            with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
                f.write(backup_content)
            time.sleep(COOLDOWN_SECONDS)
            continue

        # VRAM 安全检查：如果 LLM 提议的配置会溢出，自动降级
        current_content = read_file(TRAIN_SCRIPT)
        fixed_content, vram_msg = fix_unsafe_config(current_content)
        if fixed_content != current_content:
            with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            description += " [VRAM auto-fixed]"
        else:
            print(f"  [VRAM CHECK] {vram_msg}")

        # 提交修改
        commit_msg = f"exp{experiment_num}: {description}"
        rc, _, err = git_commit(commit_msg)
        if rc != 0:
            print(f"[WARN] Git commit 失败: {err}，但继续训练。")

        # 训练前释放 Ollama 显存
        unload_ollama_model()

        # 训练
        success, metrics, duration = run_training()
        rolling_train_sec.append(duration)

        if not success:
            print(f"[实验失败] 训练未成功完成，回滚到上一版本。")
            status = "failed"
            cds = 0.0
            # 回滚 Git 和文件
            git_reset_hard()
            with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
                f.write(backup_content)
            append_result("ROLLBACK", metrics if metrics else {}, status, description)
            last_experiments.append({"num": experiment_num, "cds": "N/A", "status": status, "desc": description})
            consecutive_discards += 1
            target_met_streak = 0
        else:
            cds = parse_cds(metrics)
            commit = git_short_hash()
            # 判断是否提升
            if cds > best_cds + CDS_KEEP_THRESHOLD:
                status = "improved"
                best_cds = cds
                best_commit = commit
                best_desc = description
                best_metrics = metrics
                consecutive_discards = 0
                # 追加结果并 amend 提交
                append_result(commit, metrics, status, description)
                git_commit_results(commit_msg)
            else:
                status = "discard"
                consecutive_discards += 1
                # 回滚
                git_reset_hard()
                with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
                    f.write(backup_content)
                append_result("DISCARD", metrics, status, description)

            # 更新目标达标连击
            if metrics_target_met(metrics):
                target_met_streak += 1
                print(f"  质量门全部 PASS，当前连击: {target_met_streak}/{TARGET_MET_STREAK_NEEDED}")
            else:
                target_met_streak = 0

            last_experiments.append({"num": experiment_num, "cds": f"{cds:.4f}", "status": status, "desc": description})

        # 更新状态面板
        update_status(
            experiment_num=experiment_num,
            best_cds=best_cds,
            best_commit=best_commit,
            best_desc=best_desc,
            metrics=best_metrics,
            target_met_streak=target_met_streak,
            consecutive_discards=consecutive_discards,
            last_experiments=last_experiments,
            what_worked=f"当前最佳描述: {best_desc} (CDS={best_cds:.4f})",
            next_experiment=f"待 LLM 生成实验 #{experiment_num+1}",
        )

        # 尝试推送（若未跳过）
        git_push()

        # 打印进度头
        print_progress_header_cn(
            phase=f"实验 #{experiment_num} 完成",
            session_start=session_start,
            exp_num=experiment_num,
            best_cds=best_cds,
            consecutive_discards=consecutive_discards,
            target_met_streak=target_met_streak,
            rolling_train_sec=list(rolling_train_sec),
        )

        # 检查停止条件
        if target_met_streak >= TARGET_MET_STREAK_NEEDED:
            print(f"\n【停止】达标连击 {target_met_streak} 次（要求 {TARGET_MET_STREAK_NEEDED}），任务完成！")
            break
        if consecutive_discards >= MAX_CONSECUTIVE_DISCARDS:
            print(f"\n【停止】连续 {consecutive_discards} 次未提升（平台期），提前结束。")
            break
        if experiment_num >= MAX_EXPERIMENTS:
            print(f"\n【停止】已达到最大实验数 {MAX_EXPERIMENTS}。")
            break

        # 冷却
        print(f"\n冷却 {COOLDOWN_SECONDS} 秒...")
        time.sleep(COOLDOWN_SECONDS)

    # ─────────────────────────────────────────────────────────────
    # 结束总结
    # ─────────────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("【自主实验结束】")
    print(f"  总实验数: {experiment_num}")
    print(f"  历史最佳 CDS: {best_cds:.4f} (commit: {best_commit})")
    print(f"  最佳描述: {best_desc}")
    print(f"  总耗时: {(time.time() - session_start) / 60:.1f} 分钟")
    print("█" * 60)

    # 最终推送
    git_push()
    print("实验记录已保存至 results.tsv 与 status.md。")


def preflight():
    """Quick sanity check: Ollama connectivity, CUDA, dataset, 1-epoch train."""
    global OLLAMA_MODEL
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "PREFLIGHT CHECK (预检)" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")

    errors = []

    # 1. Ollama connectivity
    print("\n[1/5] Ollama 连接...")
    OLLAMA_MODEL = resolve_ollama_model_name(OLLAMA_MODEL)
    test = query_ollama("Reply with exactly: OK", max_tokens=10)
    if test is None:
        errors.append("Ollama 连接失败")
        print("  ✗ 无法连接 Ollama")
    else:
        print(f"  ✓ Ollama 连接成功，模型: {OLLAMA_MODEL}")

    # 2. LLM code generation test
    print("\n[2/5] LLM 代码生成测试...")
    code_test = query_ollama(
        'You are a Python ML researcher. Reply with exactly:\nDESCRIPTION: test\nEPOCHS = 5',
        max_tokens=50
    )
    if code_test and "EPOCHS" in code_test:
        print(f"  ✓ LLM 能生成代码格式: {code_test.strip()[:80]}")
    else:
        errors.append("LLM 返回格式不正确")
        print(f"  ✗ LLM 返回格式异常: {code_test!r}")

    # 3. CUDA check
    print("\n[3/5] CUDA / GPU 检查...")
    try:
        result = subprocess.run(
            [PYTHON, "-c",
             "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), "
             "f'{torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB')"],
            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace"
        )
        out = result.stdout.strip()
        if "True" in out:
            print(f"  ✓ CUDA 可用: {out}")
        else:
            errors.append("CUDA 不可用")
            print(f"  ✗ CUDA 不可用: {out}")
    except Exception as e:
        errors.append(f"CUDA 检查失败: {e}")
        print(f"  ✗ {e}")

    # 4. Dataset & model check
    print("\n[4/5] 数据集和模型文件...")
    train_content = read_file(TRAIN_SCRIPT)
    model_match = re.search(r'^MODEL\s*=\s*["\'](.+?)["\']', train_content, re.MULTILINE)
    data_match = re.search(r'^DATA_YAML\s*=\s*["\'](.+?)["\']', train_content, re.MULTILINE)
    if model_match:
        model_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(TRAIN_SCRIPT)), model_match.group(1)))
        if os.path.isfile(model_path):
            size_mb = os.path.getsize(model_path) / 1024 / 1024
            print(f"  ✓ 模型文件: {model_path} ({size_mb:.1f}MB)")
        else:
            errors.append(f"模型文件不存在: {model_path}")
            print(f"  ✗ 模型文件不存在: {model_path}")
    if data_match:
        data_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(TRAIN_SCRIPT)), data_match.group(1)))
        if os.path.isfile(data_path):
            print(f"  ✓ 数据配置: {data_path}")
        else:
            errors.append(f"数据配置不存在: {data_path}")
            print(f"  ✗ 数据配置不存在: {data_path}")

    # VRAM estimate
    ok, vram_msg = validate_train_config(train_content)
    if ok:
        print(f"  ✓ {vram_msg}")
    else:
        errors.append(vram_msg)
        print(f"  ✗ {vram_msg}")

    # 5. Summary
    print("\n" + "═" * 60)
    if errors:
        print(f"预检发现 {len(errors)} 个问题:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\n请修复后重试。")
    else:
        print("✓ 预检全部通过！可以运行: python ollama_runner.py")
        print("\n建议启动方式（Ollama 用 CPU 推理，不占显存）:")
        print('  $env:OLLAMA_GPU_LAYERS = 0; ollama serve  # 终端 1')
        print('  python ollama_runner.py                    # 终端 2')
    print("═" * 60)


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        preflight()
    else:
        main()
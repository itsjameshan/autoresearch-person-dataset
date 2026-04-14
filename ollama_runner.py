"""
ollama_runner.py — Autonomous experiment loop driven by Ollama (local LLM).

Replaces Claude Code for machines without cloud AI access.
Uses gemma3:4b via Ollama for code generation.

Each experiment runs for a FIXED 5-MINUTE TIME BUDGET.
Expected throughput: ~12 experiments/hour, ~100 overnight.

Usage:
  1. Start Ollama (CPU mode — keep GPU free for YOLO):
       $env:OLLAMA_GPU_LAYERS = 0; ollama serve
  2. Pull model: ollama pull gemma3:4b
  3. Run: python ollama_runner.py
       Ctrl+C to stop gracefully.

Git workflow:
  - modify train.py → commit → train (5 min) → evaluate → keep/discard
  - keep: commit stays, branch advances
  - discard: git reset --hard HEAD~1, revert to best
"""

import os
import re
import sys
import json
import time
import subprocess
import datetime
import urllib.request
import urllib.error
import threading

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("AUTORESEARCH_OLLAMA_MODEL", "gemma3:4b").strip() or "gemma3:4b"
BRANCH = "autoresearch/crowd-win"

# Time budget: each experiment ~5 min training + ~2 min eval overhead = ~7 min total
# Kill at 10 min to prevent hangs
TRAIN_TIMEOUT = 600  # 10 minutes hard kill

PYTHON = sys.executable
TRAIN_SCRIPT = "train.py"
RUN_LOG = "run.log"
RESULTS_TSV = "results.tsv"
STATUS_MD = "status.md"
SUGGESTIONS_MD = "suggestions.md"
COOLDOWN_SECONDS = 10


# ══════════════════════════════════════════════════════════════
# OLLAMA
# ══════════════════════════════════════════════════════════════

def resolve_model():
    """Match requested model name against ollama list."""
    global OLLAMA_MODEL
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=15) as resp:
            names = [m["name"] for m in json.loads(resp.read().decode())["models"]]
    except Exception:
        return

    if OLLAMA_MODEL in names:
        return

    base = OLLAMA_MODEL.split(":")[0]
    for n in names:
        if n.startswith(base):
            print(f"[INFO] Model aligned: {OLLAMA_MODEL!r} → {n!r}")
            OLLAMA_MODEL = n
            return

    print(f"[WARN] {OLLAMA_MODEL!r} not found in ollama. Available: {names}")


def query_ollama(prompt, temperature=0.7, max_tokens=4096):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode()).get("response", "")
    except Exception as e:
        print(f"[ERROR] Ollama: {e}")
        return None


def unload_model():
    """Release Ollama VRAM before training."""
    try:
        payload = json.dumps({"model": OLLAMA_MODEL, "keep_alive": 0, "prompt": "", "stream": False}).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# GIT
# ══════════════════════════════════════════════════════════════

def git(*args):
    r = subprocess.run(["git"] + list(args), capture_output=True, text=True, timeout=30,
                       encoding="utf-8", errors="replace")
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def git_commit(msg):
    git("add", TRAIN_SCRIPT)
    return git("commit", "-m", msg)


def git_push():
    if os.environ.get("AUTORESEARCH_SKIP_PUSH", "").strip() in ("1", "true"):
        return 0, "", ""
    return git("push", "origin", BRANCH)


def git_short_hash():
    _, out, _ = git("rev-parse", "--short", "HEAD")
    return out


# ══════════════════════════════════════════════════════════════
# FILE OPS
# ══════════════════════════════════════════════════════════════

def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


def read_train_config():
    content = read_file(TRAIN_SCRIPT)
    m = re.search(r"# EXPERIMENT CONFIG.*?# TRAINING — do not modify", content, re.DOTALL)
    return m.group(0) if m else content[:2000]


def apply_changes(config_block):
    """Apply LLM-proposed variable assignments to train.py."""
    content = read_file(TRAIN_SCRIPT)
    valid = [
        "MODEL", "IMGSZ", "TIME_MINUTES", "BATCH", "DEVICE",
        "LR0", "LRF", "COS_LR", "HSV_H", "HSV_S", "HSV_V",
        "DEGREES", "TRANSLATE", "SCALE", "FLIPUD", "FLIPLR",
        "MOSAIC", "MIXUP", "COPY_PASTE", "ERASING", "CLOSE_MOSAIC",
        "BOX", "CLS", "AMP", "CACHE", "WORKERS", "SINGLE_CLS",
    ]
    for line in config_block.split("\n"):
        line = line.strip().replace("`", "")
        if not line or line.startswith("#"):
            continue
        for var in valid:
            if re.match(rf'^{var}\s*=', line):
                content = re.sub(rf'^{var}\s*=.*$', line, content, count=1, flags=re.MULTILINE)
                break
    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.write(content)


# ══════════════════════════════════════════════════════════════
# VRAM SAFETY
# ══════════════════════════════════════════════════════════════

def check_vram_safety():
    """Check if current train.py config will fit in 12GB VRAM."""
    content = read_file(TRAIN_SCRIPT)
    model = re.search(r'^MODEL\s*=\s*["\'](.+?)["\']', content, re.MULTILINE)
    imgsz = re.search(r'^IMGSZ\s*=\s*(\d+)', content, re.MULTILINE)
    batch = re.search(r'^BATCH\s*=\s*(\d+)', content, re.MULTILINE)
    if not (model and imgsz and batch):
        return True
    m, i, b = model.group(1).lower(), int(imgsz.group(1)), int(batch.group(1))
    # Block known-bad combos
    if "12l" in m:
        return False
    if "12s" in m and i > 640:
        return False
    if "12s" in m and b > 8:
        return False
    if i >= 1280 and b > 8:
        return False
    return True


def force_safe_config():
    """Reset to safe defaults."""
    content = read_file(TRAIN_SCRIPT)
    fixes = {"MODEL": '"person_dataset/yolo12s.pt"', "IMGSZ": "640", "BATCH": "16",
             "COPY_PASTE": "0.1", "MIXUP": "0.1"}
    for var, val in fixes.items():
        content = re.sub(rf'^{var}\s*=.*$', f'{var} = {val}', content, count=1, flags=re.MULTILINE)
    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.write(content)
    print("  [VRAM GUARD] Config reset to safe defaults")


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def run_training():
    """Run train.py with streaming output. Returns (success, duration_sec)."""
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [PYTHON, "-u", TRAIN_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, bufsize=1, encoding="utf-8", errors="replace",
    )

    killed = {"v": False}
    def _watchdog():
        time.sleep(TRAIN_TIMEOUT)
        if proc.poll() is None:
            proc.kill()
            killed["v"] = True
    threading.Thread(target=_watchdog, daemon=True).start()

    with open(RUN_LOG, "w", encoding="utf-8") as log:
        for line in (proc.stdout or []):
            log.write(line)
            log.flush()
            s = line.rstrip()
            # Stream every line to terminal in real time and keep run.log for parsing.
            print(f"  | {s}")
    proc.wait()

    dur = time.time() - t0
    if killed["v"]:
        print(f"  [TIMEOUT] Training killed after {TRAIN_TIMEOUT}s")
        return False, dur
    return proc.returncode == 0, dur


def parse_metrics():
    """Parse structured metrics from run.log (after '---' delimiter)."""
    content = read_file(RUN_LOG)
    metrics = {}
    in_metrics = False
    for line in content.split("\n"):
        line = line.strip()
        if line == "---":
            in_metrics = True
            continue
        if in_metrics and ":" in line:
            k, _, v = line.partition(":")
            metrics[k.strip()] = v.strip()
    return metrics


# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════

def init_results():
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w", encoding="utf-8") as f:
            f.write("commit\tcds\tmAP50\tmAP50_95\tsmall_obj_recall\tprecision\trecall\tmemory_gb\tstatus\tdescription\n")


def log_result(commit, metrics, status, desc):
    cds = metrics.get("cds", "0.0000")
    mAP50 = metrics.get("mAP50", "0.0000")
    mAP50_95 = metrics.get("mAP50_95", "0.0000")
    small_obj = metrics.get("small_obj_recall", "0.0000")
    prec = metrics.get("precision", "0.0000")
    rec = metrics.get("recall", "0.0000")
    mem = metrics.get("peak_memory_mb", "0")
    try:
        mem_gb = f"{float(mem)/1024:.1f}"
    except (ValueError, TypeError):
        mem_gb = "0.0"
    row = f"{commit}\t{cds}\t{mAP50}\t{mAP50_95}\t{small_obj}\t{prec}\t{rec}\t{mem_gb}\t{status}\t{desc}\n"
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(row)


def update_status(exp_num, best_cds, best_commit, best_desc, last_exps):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    table = ""
    for e in last_exps[-5:]:
        table += f"| {e['num']} | {e['cds']} | {e['status']} | {e['desc'][:60]} |\n"
    content = f"""# Autoresearch Status

## Session Info
- Updated: {now}
- Branch: {BRANCH}
- Total experiments: {exp_num}
- Platform: Windows / RTX 5070 12GB / Ollama {OLLAMA_MODEL}
- Time budget: 5 min/experiment

## Current Best
- CDS: {best_cds:.4f} (commit: {best_commit})
- Description: {best_desc}

## Last 5 Experiments
| # | CDS | Status | Description |
|---|-----|--------|-------------|
{table}"""
    with open(STATUS_MD, "w", encoding="utf-8") as f:
        f.write(content)


# ══════════════════════════════════════════════════════════════
# LLM EXPERIMENT GENERATION
# ══════════════════════════════════════════════════════════════

def generate_experiment(exp_num, config, history, suggestions):
    prompt = f"""You are an autonomous ML researcher optimizing YOLO for dense crowd detection.
Goal: maximize CDS (Crowd Detection Score). Higher is better.

Hardware: RTX 5070 12GB VRAM, 64GB RAM. Each experiment has a FIXED 5-MINUTE time budget.

=== VRAM CONSTRAINTS (MUST OBEY) ===
- 12GB VRAM limit. Exceeding causes 10x slowdown.
- Safe: yolov8s+640+batch16, yolov8s+1280+batch8, yolo12s+640+batch8
- UNSAFE (BANNED): yolo12s+1280, yolo12l, copy_paste>0.2, mixup>0.2
- Model paths: "person_dataset/yolov8s.pt", "person_dataset/yolo12s.pt", etc.
- Do NOT change DEVICE, DATA_YAML, CACHE, WORKERS, TIME_MINUTES

Current train.py config:
{config}

Experiment history (most recent last):
{history}

Suggestions:
{suggestions}

Experiment #{exp_num}. Propose ONE focused change. Think about what will improve CDS the most.

Reply EXACTLY:
DESCRIPTION: <what and why, one line>
VAR = value
"""
    resp = query_ollama(prompt, temperature=0.7)
    if not resp:
        return None, None

    desc = ""
    config_lines = []
    for line in resp.strip().split("\n"):
        line = line.strip()
        if line.startswith("DESCRIPTION:"):
            desc = line.replace("DESCRIPTION:", "").strip()
        elif re.match(r'^[A-Z_0-9]+\s*=', line):
            config_lines.append(line.replace("`", ""))
    return desc or f"experiment {exp_num}", "\n".join(config_lines)


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    global OLLAMA_MODEL
    print("=" * 60)
    print(" AUTORESEARCH — 5-min time budget, Ollama-driven")
    print(f" Model: {OLLAMA_MODEL} | Branch: {BRANCH}")
    print("=" * 60)

    # Resolve Ollama model name
    resolve_model()
    test = query_ollama("Say OK", max_tokens=10)
    if test is None:
        print("[FATAL] Cannot connect to Ollama. Exiting.")
        sys.exit(1)
    print(f"[OK] Ollama connected: {OLLAMA_MODEL}")

    # Git setup
    git("checkout", BRANCH)
    init_results()
    suggestions = read_file(SUGGESTIONS_MD)

    # State
    best_cds = -1.0
    best_commit = ""
    best_desc = ""
    last_exps = []
    session_start = time.time()

    # ── Baseline (experiment #0) ──
    print("\n>>> BASELINE (no modifications) <<<")
    unload_model()
    if not check_vram_safety():
        force_safe_config()

    git_commit("baseline")
    success, dur = run_training()
    commit = git_short_hash()

    if success:
        metrics = parse_metrics()
        cds = float(metrics.get("cds", 0))
        best_cds, best_commit, best_desc = cds, commit, "baseline"
        log_result(commit, metrics, "keep", "baseline")
        last_exps.append({"num": 0, "cds": f"{cds:.4f}", "status": "keep", "desc": "baseline"})
        git("add", RESULTS_TSV, STATUS_MD)
        git("commit", "--amend", "--no-edit")
        git_push()
        print(f"\n>>> BASELINE CDS: {cds:.4f} ({dur:.0f}s) <<<")
    else:
        print("[FATAL] Baseline failed. Check train.py and environment.")
        log_result(commit, {}, "crash", "baseline crash")
        sys.exit(1)

    # ── Experiment loop ──
    exp_num = 0
    while True:
        exp_num += 1
        elapsed = (time.time() - session_start) / 3600
        print(f"\n{'#' * 60}")
        print(f"# Experiment #{exp_num} | Best CDS: {best_cds:.4f} | Session: {elapsed:.1f}h")
        print(f"{'#' * 60}")

        # Ask LLM
        config = read_train_config()
        history = read_file(RESULTS_TSV)
        desc, changes = generate_experiment(exp_num, config, history, suggestions)

        if not changes:
            print("[WARN] LLM returned no changes, retrying...")
            desc, changes = generate_experiment(exp_num, config, history, suggestions)
            if not changes:
                print("[SKIP] No valid changes after retry.")
                continue

        print(f"  Hypothesis: {desc}")
        for line in changes.split("\n"):
            if line.strip():
                print(f"    {line.strip()}")

        # Save backup, apply changes
        backup = read_file(TRAIN_SCRIPT)
        apply_changes(changes)

        # VRAM safety check
        if not check_vram_safety():
            print("  [VRAM GUARD] Unsafe config, reverting to safe defaults")
            force_safe_config()
            desc += " [VRAM-fixed]"

        # Commit → train → evaluate
        git_commit(f"exp{exp_num}: {desc}")
        unload_model()
        success, dur = run_training()
        commit = git_short_hash()

        if success:
            metrics = parse_metrics()
            cds = float(metrics.get("cds", 0))
            improvement = cds - best_cds

            if improvement > 0.001:
                # KEEP — branch advances
                status = "keep"
                best_cds, best_commit, best_desc = cds, commit, desc
                log_result(commit, metrics, status, desc)
                git("add", RESULTS_TSV, STATUS_MD)
                git("commit", "--amend", "--no-edit")
                git_push()
                print(f"  >>> KEEP — CDS {cds:.4f} (+{improvement:.4f})")
            else:
                # DISCARD — revert
                status = "discard"
                log_result(commit, metrics, status, desc)
                git("reset", "--hard", "HEAD~1")
                # Commit just the results log
                git("add", RESULTS_TSV)
                git("commit", "-m", f"log: discard — {desc[:60]}")
                git_push()
                print(f"  >>> DISCARD — CDS {cds:.4f} ({improvement:+.4f})")
        else:
            # CRASH — revert
            status = "crash"
            cds = 0.0
            log_result("CRASH", {}, status, desc)
            git("reset", "--hard", "HEAD~1")
            with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
                f.write(backup)
            git("add", RESULTS_TSV)
            git("commit", "-m", f"log: crash — {desc[:60]}")
            print(f"  >>> CRASH — reverted")

        last_exps.append({"num": exp_num, "cds": f"{cds:.4f}" if cds else "CRASH", "status": status, "desc": desc})
        update_status(exp_num, best_cds, best_commit, best_desc, last_exps)

        print(f"  Duration: {dur:.0f}s | Cooling {COOLDOWN_SECONDS}s...")
        time.sleep(COOLDOWN_SECONDS)


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        print("=" * 60)
        print(" PREFLIGHT CHECK")
        print("=" * 60)
        resolve_model()
        test = query_ollama("Reply: OK", max_tokens=10)
        print(f"  Ollama: {'OK' if test else 'FAIL'} ({OLLAMA_MODEL})")

        r = subprocess.run([PYTHON, "-c",
            "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"],
            capture_output=True, text=True, timeout=30)
        print(f"  CUDA: {r.stdout.strip()}")

        content = read_file(TRAIN_SCRIPT)
        print(f"  VRAM safe: {check_vram_safety()}")

        model = re.search(r'^MODEL\s*=\s*["\'](.+?)["\']', content, re.MULTILINE)
        if model:
            p = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(TRAIN_SCRIPT)), model.group(1)))
            print(f"  Model file: {p} ({'EXISTS' if os.path.isfile(p) else 'MISSING'})")
        print("=" * 60)
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\n[Ctrl+C] Stopped. Results saved in results.tsv and status.md.")

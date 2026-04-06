"""
ollama_runner.py — Autonomous experiment loop driven by Ollama (local LLM).

Replaces Claude Code / Codex CLI for machines without cloud AI access.
Uses Ollama API (http://localhost:11434) with qwen2.5-coder:14b.

Usage:
  1. Start Ollama: ollama serve
  2. Pull model: ollama pull qwen2.5-coder:14b
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

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:14b"
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
    """Run train.py and return (success, metrics_dict)."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {PYTHON} {TRAIN_SCRIPT}")
    print(f"{'='*60}\n")

    with open(RUN_LOG, "w") as log:
        try:
            proc = subprocess.run(
                [PYTHON, TRAIN_SCRIPT],
                stdout=log, stderr=subprocess.STDOUT,
                timeout=TRAIN_TIMEOUT,
            )
            success = proc.returncode == 0
        except subprocess.TimeoutExpired:
            print("TIMEOUT: Training exceeded 60 minutes!")
            return False, {}

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
                       "peak_memory_mb", "epochs_completed",
                       "precision_gate", "recall_gate"):
                metrics[key] = val

    if "cds" not in metrics and success:
        # Check if training completed but eval failed
        if "Error" in log_content or "Traceback" in log_content:
            success = False

    return success, metrics


def parse_cds(metrics):
    """Extract CDS as float from metrics dict."""
    try:
        return float(metrics.get("cds", 0))
    except (ValueError, TypeError):
        return 0.0


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
    mem = metrics.get("peak_memory_mb", "0.0")
    epochs = metrics.get("epochs_completed", "0")

    try:
        mem_gb = f"{float(mem) / 1024:.1f}"
    except (ValueError, TypeError):
        mem_gb = "0.0"

    row = f"{commit}\t{cds}\t{mAP50}\t{mAP50_95}\t{small_obj}\t{precision}\t{recall}\t{mae}\t{inf_ms}\t{mem_gb}\t{epochs}\t{status}\t{description}\n"

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
- TARGET_MET streak: {target_met_streak}/{TARGET_MET_STREAK_NEEDED} needed to stop

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
Your goal: maximize CDS (Crowd Detection Score), targeting precision>=0.90 and recall>=0.85.

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
    print("=" * 60)
    print("AUTORESEARCH — Ollama-Driven Autonomous Experiment Loop")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Branch: {BRANCH}")
    print(f"Max experiments: {MAX_EXPERIMENTS}")
    print("=" * 60)

    # Check Ollama connectivity
    test = query_ollama("Say OK", max_tokens=10)
    if test is None:
        print("\nFATAL: Cannot connect to Ollama. Exiting.")
        sys.exit(1)
    print(f"Ollama connected: {OLLAMA_MODEL}")

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
                    best_desc = parts[-1] if len(parts) >= 13 else "previous"
            except ValueError:
                pass

    # ── BASELINE (if no results yet) ──
    if start_num == 0:
        print("\n>>> RUNNING BASELINE (no modifications to train.py) <<<\n")
        git_commit("experiment: baseline — " + read_train_config()[:80].replace("\n", " "))
        success, metrics = run_training()
        commit = git_short_hash()

        if success and "cds" in metrics:
            cds = parse_cds(metrics)
            best_cds = cds
            best_commit = commit
            best_desc = "baseline"
            target_met = "PASS" in metrics.get("precision_gate", "") and "PASS" in metrics.get("recall_gate", "")

            append_result(commit, metrics, "keep", "baseline" + (" [TARGET_MET]" if target_met else ""))
            git("add", RESULTS_TSV, STATUS_MD)
            git("commit", "--amend", "--no-edit")
            git_push()
            print(f"\nBASELINE CDS: {cds}")
            last_experiments.append({"num": 0, "cds": cds, "status": "keep", "desc": "baseline"})
        else:
            print("BASELINE FAILED!")
            tail = read_file(RUN_LOG)[-500:]
            print(tail)
            append_result(commit, metrics, "crash", "baseline crash")
            git_reset_hard()
            # Still continue — maybe the config needs adjustment

        start_num = 1

    # ── EXPERIMENT LOOP ──
    for exp_num in range(start_num, MAX_EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT #{exp_num}")
        print(f"Best CDS: {best_cds} | Discards: {consecutive_discards} | Target streak: {target_met_streak}")
        print(f"{'='*60}")

        # Check stop conditions
        if target_met_streak >= TARGET_MET_STREAK_NEEDED:
            print(f"\n>>> TARGET MET {TARGET_MET_STREAK_NEEDED}x CONSECUTIVE — STOPPING <<<")
            git_push()
            break

        if consecutive_discards >= MAX_CONSECUTIVE_DISCARDS:
            print(f"\n>>> {MAX_CONSECUTIVE_DISCARDS} CONSECUTIVE DISCARDS — PLATEAU — STOPPING <<<")
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
        current_config = read_train_config()
        results_history = get_results_history()
        suggestions = read_file(SUGGESTIONS_MD)

        description, config_block = generate_experiment(
            exp_num, current_config, results_history, suggestions
        )

        if not config_block:
            print("WARNING: Ollama returned no config changes. Retrying with higher temperature...")
            description, config_block = generate_experiment(
                exp_num, current_config, results_history, suggestions
            )
            if not config_block:
                print("Skipping this experiment.")
                consecutive_discards += 1
                continue

        # Apply changes
        print(f"Hypothesis: {description}")
        print(f"Changes:\n{config_block}")
        apply_train_changes(config_block)

        # Git commit
        git_commit(f"experiment: {description}")

        # Run training
        success, metrics = run_training()
        commit = git_short_hash()

        if not success or "cds" not in metrics:
            # CRASH
            print("CRASH — checking error...")
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
            target_met = "PASS" in metrics.get("precision_gate", "") and "PASS" in metrics.get("recall_gate", "")

            print(f"CDS: {cds} (best: {best_cds}, diff: {improvement:+.4f})")

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
                print(f">>> KEEP — CDS improved by {improvement:+.4f}")

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
                print(f">>> DISCARD — CDS change: {improvement:+.4f}")

                last_experiments.append({"num": exp_num, "cds": cds, "status": "discard", "desc": description})

        # Update status dashboard
        update_status(
            exp_num + 1, best_cds, best_commit, best_desc,
            metrics, target_met_streak, consecutive_discards,
            last_experiments, what_worked,
            "generating next hypothesis..."
        )

        # Cooldown
        print(f"Cooling down {COOLDOWN_SECONDS}s...")
        time.sleep(COOLDOWN_SECONDS)

    # Final summary
    print("\n" + "=" * 60)
    print("AUTORESEARCH SESSION COMPLETE")
    print(f"Total experiments: {len(last_experiments)}")
    print(f"Best CDS: {best_cds} (commit: {best_commit})")
    print(f"Description: {best_desc}")
    print("=" * 60)

    # Final push
    git("add", RESULTS_TSV, STATUS_MD)
    git("commit", "-m", "autoresearch: session complete")
    git_push()


if __name__ == "__main__":
    main()

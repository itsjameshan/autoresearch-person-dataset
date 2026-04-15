"""
ollama_runner.py — YOLO12l 专用自动优化
固定：YOLO12l + 1280尺寸 + batch=4 + qwen2.5-coder:7b
自动停止：达到指标阈值 / 30轮 / 5轮不提升
自动保存最优模型权重
最终自动保存【效果最好的一轮】权重到 best_final_model
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
import shutil

# ══════════════════════════════════════════════════════════════
# 固定配置（按你的硬件与需求）
# ══════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
BRANCH = "autoresearch/crowd-win"

TRAIN_TIMEOUT = 600
PYTHON = sys.executable
TRAIN_SCRIPT = "train.py"
RUN_LOG = "run.log"
RESULTS_TSV = "results.tsv"
STATUS_MD = "status.md"

BEST_PT_DIR = "best_model"
FINAL_BEST_DIR = "best_final_model"  # 最终最优模型单独保存

MAX_EXPERIMENTS = 300
MAX_NO_IMPROVE = 10
STOP_CDS = 0.95
STOP_MAP50 = 0.92

# ══════════════════════════════════════════════════════════════
# OLLAMA 交互
# ══════════════════════════════════════════════════════════════

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
    try:
        payload = json.dumps({"model": OLLAMA_MODEL, "keep_alive": 0}).encode()
        urllib.request.urlopen(urllib.request.Request(OLLAMA_URL, data=payload), timeout=10).read()
    except:
        pass

# ══════════════════════════════════════════════════════════════
# Git 操作
# ══════════════════════════════════════════════════════════════

def git(*args):
    r = subprocess.run(["git"] + list(args), capture_output=True, text=True, timeout=20)
    return r.returncode, r.stdout.strip(), r.stderr.strip()

def git_commit(msg):
    git("add", TRAIN_SCRIPT)
    return git("commit", "-m", msg)

def git_short_hash():
    _, out, _ = git("rev-parse", "--short", "HEAD")
    return out

# ══════════════════════════════════════════════════════════════
# 文件操作
# ══════════════════════════════════════════════════════════════

def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def apply_changes(config_block):
    content = read_file(TRAIN_SCRIPT)
    valid_params = [
        "LR0", "LRF", "MOSAIC", "MIXUP", "COPY_PASTE",
        "DEGREES", "TRANSLATE", "SCALE", "BOX", "CLS",
        "CONF", "IOU"  # 增加置信度 & NMS 优化
    ]
    changes_made = []
    for line in config_block.split("\n"):
        line = line.strip().replace("`", "")
        for param in valid_params:
            if re.match(rf"^{param}\s*=", line):
                content = re.sub(rf"^{param}\s*=.*", line, content, flags=re.MULTILINE)
                changes_made.append(line)
    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.write(content)
    return changes_made

# ══════════════════════════════════════════════════════════════
# 强制 YOLO12l + 1280 + batch=4
# ══════════════════════════════════════════════════════════════

def force_yolo12l_config():
    content = read_file(TRAIN_SCRIPT)
    fixed_config = {
        "MODEL": '"person_dataset/yolo12l.pt"',
        "IMGSZ": "1280",
        "BATCH": "4",
        "TIME_MINUTES": "5",
        "DEVICE": "0",
        "SINGLE_CLS": "True"
    }
    for key, value in fixed_config.items():
        content = re.sub(rf"^{key}\s*=.*", f"{key} = {value}", content, flags=re.MULTILINE)
    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.write(content)

# ══════════════════════════════════════════════════════════════
# 训练执行（日志追加，不覆盖）
# ══════════════════════════════════════════════════════════════

def run_training(exp_num):
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [PYTHON, "-u", TRAIN_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, bufsize=1, encoding="utf-8", errors="replace"
    )

    def watchdog():
        time.sleep(TRAIN_TIMEOUT)
        if proc.poll() is None:
            proc.kill()
    threading.Thread(target=watchdog, daemon=True).start()

    with open(RUN_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"EXPERIMENT {exp_num} START | {time.ctime()}\n")
        log_file.write(f"{'='*80}\n")
        log_file.flush()
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            print(f"  | {line.rstrip()}")
    proc.wait()
    return proc.returncode == 0, time.time() - t0

# ══════════════════════════════════════════════════════════════
# 解析指标（含 Recall / Precision / mAP50）
# ══════════════════════════════════════════════════════════════

def parse_metrics():
    log = read_file(RUN_LOG)
    metrics = {"cds":0.0, "mAP50":0.0, "Recall":0.0, "Precision":0.0}
    for key in metrics:
        match = re.search(rf"{key}\s*[:=]\s*([0-9\.]+)", log)
        if match:
            metrics[key] = float(match.group(1))
    return metrics

# ══════════════════════════════════════════════════════════════
# 保存最优权重
# ══════════════════════════════════════════════════════════════

def save_best_weights():
    os.makedirs(BEST_PT_DIR, exist_ok=True)
    for root, dirs, files in os.walk("autoresearch_runs/current/weights"):
        if "best.pt" in files:
            src = os.path.join(root, "best.pt")
            dst = os.path.join(BEST_PT_DIR, "best.pt")
            shutil.copy2(src, dst)
            print(f"✅ 最优权重保存：{dst}")
            return

# ══════════════════════════════════════════════════════════════
# 【新增】最终保存全局最优模型
# ══════════════════════════════════════════════════════════════

def save_final_best_model():
    os.makedirs(FINAL_BEST_DIR, exist_ok=True)
    src = os.path.join(BEST_PT_DIR, "best.pt")
    dst = os.path.join(FINAL_BEST_DIR, "best.pt")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"\n🎉 【最终全局最优模型已保存】：{dst}")
        print(f"📁 路径：best_final_model/best.pt")
    else:
        print("\n❌ 未找到最优模型")

# ══════════════════════════════════════════════════════════════
# 记录结果
# ══════════════════════════════════════════════════════════════

def init_results():
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w", encoding="utf-8") as f:
            f.write("exp_num\tcds\tmAP50\tRecall\tPrecision\tstatus\toptimized_params\n")

def log_experiment_result(exp_num, m, status, params):
    p = " | ".join(params) if params else "none"
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(f"{exp_num}\t{m['cds']:.4f}\t{m['mAP50']:.4f}\t{m['Recall']:.4f}\t{m['Precision']:.4f}\t{status}\t{p}\n")

# ══════════════════════════════════════════════════════════════
# LLM生成优化（含 conf / iou 优化）
# ══════════════════════════════════════════════════════════════

def generate_optimization(exp_num, current_config, history):
    prompt = f"""你是专业YOLO12l调参专家，专注体育场观众计数。
优化目标：
1. 高mAP50
2. 高Recall（不漏检）
3. 高Precision（不误检）
4. 最优置信度CONF & 最优NMS_IOU
只输出参数，不要多余内容。
允许优化：LR0,LRF,MOSAIC,MIXUP,COPY_PASTE,DEGREES,TRANSLATE,SCALE,BOX,CLS,CONF,IOU
严禁修改：MODEL,IMGSZ,BATCH,DEVICE,TIME_MINUTES

当前参数：
{current_config}

历史结果：
{history}

输出格式：
DESCRIPTION: ...
参数=值
"""
    res = query_ollama(prompt, temperature=0.6)
    if not res:
        return None, None
    desc = ""
    lines = []
    for line in res.split("\n"):
        line = line.strip()
        if line.startswith("DESCRIPTION:"):
            desc = line.replace("DESCRIPTION:", "").strip()
        elif "=" in line and not line.startswith("#"):
            lines.append(line.replace("`", ""))
    return desc or f"Exp {exp_num}", "\n".join(lines)

# ══════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("YOLO12l 自动调参训练系统 | 最优置信度+NMS | 不漏检不误检")
    print("最终最优模型将保存到：best_final_model/best.pt")
    print("=" * 80)

    force_yolo12l_config()
    init_results()

    best = {"cds":0.0, "mAP50":0.0, "Recall":0.0, "Precision":0.0}
    no_improve = 0
    exp = 0

    try:
        while exp < MAX_EXPERIMENTS:
            exp += 1
            print(f"\n【第 {exp}/{MAX_EXPERIMENTS} 轮】")
            print(f"最佳：CDS={best['cds']:.4f} | mAP50={best['mAP50']:.4f} | Recall={best['Recall']:.4f} | Precision={best['Precision']:.4f}")

            current = read_file(TRAIN_SCRIPT)
            history = read_file(RESULTS_TSV)
            desc, changes = generate_optimization(exp, current, history)

            if not changes:
                print("⚠️ 未生成参数，跳过")
                continue

            params = apply_changes(changes)
            print(f"策略：{desc}")
            print(f"参数：{params}")

            git_commit(f"exp{exp}: {desc[:50]}")
            unload_model()
            ok, duration = run_training(exp)

            m = parse_metrics() if ok else {"cds":0, "mAP50":0, "Recall":0, "Precision":0}
            status = "DISCARD"

            current_score = m["cds"] + m["Recall"] + m["Precision"]
            best_score = best["cds"] + best["Recall"] + best["Precision"]

            if current_score > best_score + 0.001:
                best = m.copy()
                no_improve = 0
                status = "BEST"
                save_best_weights()
                print("✅ 刷新最优！")
            else:
                no_improve += 1
                git("reset", "--hard", "HEAD~1")
                print(f"❌ 无提升，连续{no_improve}轮")

            log_experiment_result(exp, m, status, params)

            if no_improve >= MAX_NO_IMPROVE:
                print("\n🛑 连续无提升，停止训练")
                break
            if best["cds"] >= STOP_CDS or best["mAP50"] >= STOP_MAP50:
                print("\n🛑 达到目标精度，停止训练")
                break

    except KeyboardInterrupt:
        print("\n⏹️ 手动停止")

    # ✅ 最后保存全局最优模型
    save_final_best_model()

    print("\n" + "="*50)
    print("🎉 训练全部完成！")
    print("最终最优指标：")
    print(f"CDS: {best['cds']:.4f}")
    print(f"mAP50: {best['mAP50']:.4f}")
    print(f"Recall: {best['Recall']:.4f}")
    print(f"Precision: {best['Precision']:.4f}")
    print(f"最优模型：best_final_model/best.pt")
    print("="*50)

if __name__ == "__main__":
    main()
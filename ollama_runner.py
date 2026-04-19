"""
ollama_runner.py - YOLO12s 专用自动优化
固定：YOLO12s + 1280尺寸 + batch=2 + qwen2.5-coder:7b
自动停止：达到指标阈值 / 30轮 / 5轮不提升
自动保存最优模型权重
最终保存效果最好的一轮权重到 best_final_model
"""

import os
import re
import sys
import json
import time
import subprocess
import urllib.request
import urllib.error
import threading
import shutil

# ======================================================================
# 固定配置
# ======================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
BRANCH = "autoresearch/crowd-win"

TRAIN_TIMEOUT = 3600
PYTHON = sys.executable
TRAIN_SCRIPT = "train.py"
RUN_LOG = "run.log"
RESULTS_TSV = "results.tsv"
STATUS_MD = "status.md"

BEST_PT_DIR = "best_model"
FINAL_BEST_DIR = "best_final_model"

MAX_EXPERIMENTS = 300
MAX_NO_IMPROVE = 10
STOP_CDS = 0.95
STOP_MAP50 = 0.92

# ======================================================================
# OLLAMA 交互
# ======================================================================

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
        print("[ERROR] Ollama: {}".format(e))
        return None

def unload_model():
    try:
        payload = json.dumps({"model": OLLAMA_MODEL, "keep_alive": 0}).encode()
        urllib.request.urlopen(urllib.request.Request(OLLAMA_URL, data=payload), timeout=10).read()
    except:
        pass


def git(*args):
    try:
        r = subprocess.run(["git"] + list(args), capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except:
        return -1, "", ""

def git_commit(msg):
    return

def git_short_hash():
    return ""

# ======================================================================
# 文件操作
# ======================================================================

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
        "CONF", "IOU"
    ]
    changes_made = []
    for line in config_block.split("\n"):
        line = line.strip().replace("`", "")
        for param in valid_params:
            if re.match(rf"^{param}\s*=.*", line):
                content = re.sub(rf"^{param}\s*=.*", line, content, flags=re.MULTILINE)
                changes_made.append(line)
    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.write(content)
    return changes_made

# ======================================================================
# 强制固定配置：YOLO12s
# ======================================================================

def force_yolo12l_config():
    content = read_file(TRAIN_SCRIPT)
    fixed_config = {
        "MODEL": '"person_dataset/yolo12s.pt"',
        "IMGSZ": "1280",
        "BATCH": "2",
        "TIME_MINUTES": "0",
        "DEVICE": "0",
        "SINGLE_CLS": "True"
    }
    for key, value in fixed_config.items():
        content = re.sub(rf"^{key}\s*=.*", f"{key} = {value}", content, flags=re.MULTILINE)
    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.write(content)

# ======================================================================
# 训练执行
# ======================================================================

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
        log_file.write("\n" + "="*80 + "\n")
        log_file.write("EXPERIMENT {} START | {}\n".format(exp_num, time.ctime()))
        log_file.write("="*80 + "\n")
        log_file.flush()
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            print("  | {}".format(line.rstrip()))
    proc.wait()
    return proc.returncode == 0, time.time() - t0

# ======================================================================
# 解析指标（已修复 100% 适配你的日志格式）
# ======================================================================

def parse_metrics():
    log = read_file(RUN_LOG)
    metrics = {"cds": 0.0, "mAP50": 0.0, "Recall": 0.0, "Precision": 0.0}
    pattern = re.compile(r"all\s+\d+\s+\d+\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)")
    matches = pattern.findall(log)

    if matches:
        p, r, m50, m = matches[-1]
        metrics["Precision"] = float(p)
        metrics["Recall"] = float(r)
        metrics["mAP50"] = float(m50)
        metrics["cds"] = float(m50)  # CDS 复用 mAP50

    return metrics
    if match:
        metrics["Precision"] = float(match.group(1))
        metrics["Recall"]    = float(match.group(2))
        metrics["mAP50"]     = float(match.group(3))
        metrics["cds"]       = float(match.group(3))
        print(f"成功解析指标：Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, mAP50={metrics['mAP50']:.4f}")
        return metrics
    else:
        print("未找到指标，使用默认值")
        return metrics

# ======================================================================
# 保存最优模型
# ======================================================================

def save_best_weights():
    os.makedirs(BEST_PT_DIR, exist_ok=True)
    for root, dirs, files in os.walk("autoresearch_runs/current/weights"):
        if "best.pt" in files:
            src = os.path.join(root, "best.pt")
            dst = os.path.join(BEST_PT_DIR, "best.pt")
            shutil.copy2(src, dst)
            print("最优权重保存：{}".format(dst))
            return

def save_final_best_model():
    os.makedirs(FINAL_BEST_DIR, exist_ok=True)
    src = os.path.join(BEST_PT_DIR, "best.pt")
    dst = os.path.join(FINAL_BEST_DIR, "best.pt")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print("\n最终全局最优模型已保存：{}".format(dst))
        print("路径：best_final_model/best.pt")
    else:
        print("\n未找到最优模型")

# ======================================================================
# 记录结果
# ======================================================================

def init_results():
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w", encoding="utf-8") as f:
            f.write("exp_num\tcds\tmAP50\tRecall\tPrecision\tstatus\toptimized_params\n")

def log_experiment_result(exp_num, m, status, params):
    p = " | ".join(params) if params else "none"
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(f"{exp_num}\t{m['cds']:.4f}\t{m['mAP50']:.4f}\t{m['Recall']:.4f}\t{m['Precision']:.4f}\t{status}\t{p}\n")

# ======================================================================
# 生成优化参数
# ======================================================================

def generate_optimization(exp_num, current_config, history):
    prompt = f"""你是专业YOLO12s调参专家，专注体育场观众计数。
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
    return desc or "Exp {}".format(exp_num), "\n".join(lines)

# ======================================================================
# 主循环
# ======================================================================

def main():
    print("=" * 80)
    print("YOLO12s 自动调参训练系统 | 最优置信度+NMS | 不漏检不误检")
    print("最终最优模型将保存到：best_final_model/best.pt")
    print("=" * 80)

    force_yolo12l_config()
    init_results()

    best = {"cds": 0.0, "mAP50": 0.0, "Recall": 0.0, "Precision": 0.0}
    no_improve = 0
    exp = 0

    try:
        while exp < MAX_EXPERIMENTS:
            exp += 1
            print("\n【第 {}/{} 轮】".format(exp, MAX_EXPERIMENTS))
            print("最佳：CDS={:.4f} | mAP50={:.4f} | Recall={:.4f} | Precision={:.4f}".format(
                best['cds'], best['mAP50'], best['Recall'], best['Precision']))

            current = read_file(TRAIN_SCRIPT)
            history = read_file(RESULTS_TSV)
            desc, changes = generate_optimization(exp, current, history)

            if not changes:
                print("未生成参数，跳过")
                continue

            params = apply_changes(changes)
            print("策略：{}".format(desc))
            print("参数：{}".format(params))

            unload_model()
            ok, duration = run_training(exp)

            m = parse_metrics()
            status = "DISCARD"

            current_score = m["cds"] + m["Recall"] + m["Precision"]
            best_score = best["cds"] + best["Recall"] + best["Precision"]

            print(f"\n====== 本轮训练结果 ======")
            print(
                f"本轮指标：CDS={m['cds']:.4f} | mAP50={m['mAP50']:.4f} | Recall={m['Recall']:.4f} | Precision={m['Precision']:.4f}")
            print(f"当前总分：{current_score:.4f} | 历史最佳：{best_score:.4f}")

            if current_score > best_score + 0.001:
                print("判断：本轮有提升，刷新最优！")
            else:
                print("判断：本轮无提升，继续优化！")

            if current_score > best_score + 0.001:
                best = m.copy()
                no_improve = 0
                status = "BEST"
                save_best_weights()
                print("刷新最优！")
            else:
                no_improve += 1
                print("无提升，连续{}轮".format(no_improve))

            log_experiment_result(exp, m, status, params)

            if no_improve >= MAX_NO_IMPROVE:
                print("\n连续无提升，停止训练")
                break
            if best["cds"] >= STOP_CDS or best["mAP50"] >= STOP_MAP50:
                print("\n达到目标精度，停止训练")
                break

    except KeyboardInterrupt:
        print("\n手动停止")

    save_final_best_model()

    print("\n" + "=" * 50)
    print("训练全部完成！")
    print("最终最优指标：")
    print("CDS: {:.4f}".format(best['cds']))
    print("mAP50: {:.4f}".format(best['mAP50']))
    print("Recall: {:.4f}".format(best['Recall']))
    print("Precision: {:.4f}".format(best['Precision']))
    print("最优模型：best_final_model/best.pt")
    print("=" * 50)

if __name__ == "__main__":
    main()
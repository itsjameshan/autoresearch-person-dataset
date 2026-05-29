import os
import re
import sys
import json
import time
import shutil
import subprocess
import threading
import torch
from skopt import Optimizer
from skopt.space import Real

# ==============================================
# 【你必须看懂的配置】
# ==============================================
TRAIN_TIMEOUT       = 7200        # 安全超时：2小时（绝对够用）
PYTHON              = sys.executable
TRAIN_SCRIPT        = "train.py"
LOG_FILE            = "run.log"   # 训练日志文件
RESULTS_FILE        = "results.tsv" # 结果表格文件

MAX_EXPERIMENTS     = 25          # 贝叶斯最多搜索25组参数（1天多跑完）
MAX_NO_IMPROVE      = 8           # 连续8组不涨分就停止
TARGET_MAP50        = 0.92        # 达到这个分数直接停止
TRAIN_EPOCHS        = 2           # 【核心修改】每组参数跑2个epoch

# 贝叶斯优化的10个参数（搜索范围）
search_space = [
    Real(0.0005, 0.0035,  name="LR0"),
    Real(0.00005, 0.0005, name="LRF"),
    Real(0.0005, 0.003,   name="CONF"),
    Real(0.4, 0.65,       name="IOU"),
    Real(10.0, 30.0,      name="BOX"),
    Real(0.5, 1.5,        name="CLS"),
    Real(0.3, 0.7,        name="MOSAIC"),
    Real(0.0, 0.2,        name="MIXUP"),
    Real(0.0, 0.2,        name="COPY_PASTE"),
    Real(5.0, 20.0,       name="DEGREES"),
]

optimizer = Optimizer(dimensions=search_space, base_estimator="GP", acq_func="gp_hedge", random_state=42)

# ==============================================
# 初始化结果文件（只写一次表头）
# ==============================================
def init_results_file():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("exp\tmAP50\tRecall\tPrecision\tscore\tstatus\n")

# ==============================================
# 写入单轮结果（追加模式，不会覆盖）
# ==============================================
def append_result(exp_num, m50, r, p, score, status):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{exp_num}\t{m50:.4f}\t{r:.4f}\t{p:.4f}\t{score:.4f}\t{status}\n")

# ==============================================
# 训练函数（固定跑2个epoch）
# ==============================================
def run_training(exp_num):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 每次训练前清空旧日志，避免解析到上一轮的结果
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # 启动训练，固定跑2个epoch
    proc = subprocess.Popen(
        [PYTHON, "-u", TRAIN_SCRIPT, str(TRAIN_EPOCHS)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    def watchdog():
        time.sleep(TRAIN_TIMEOUT)
        try:
            proc.kill()
        except:
            pass
    threading.Thread(target=watchdog, daemon=True).start()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            print(line.rstrip())
    proc.wait()
    return proc.returncode == 0

# ==============================================
# 从日志读取 mAP / Recall / Precision
# ==============================================
def parse_results():
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        pattern = re.compile(r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
        matches = pattern.findall(txt)
        if matches:
            p, r, m50, m = matches[-1]
            return float(m50), float(r), float(p)
    except:
        pass
    return 0.0, 0.0, 0.0

# ==============================================
# 把参数写入train.py
# ==============================================
def _sanitize(val):
    """Produce a valid Python literal for train.py assignment."""
    if isinstance(val, bool):
        return repr(val)
    if isinstance(val, (int, float)):
        if isinstance(val, float):
            import math
            if math.isnan(val):
                return "float('nan')"
            if math.isinf(val):
                return "float('-inf')" if val < 0 else "float('inf')"
        return repr(val)
    if isinstance(val, str):
        return repr(val)
    return repr(float(val))


def apply_params(param_dict):
    with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        for key, val in param_dict.items():
            if re.match(rf"^{key}\s*=.*", line.strip()):
                lines[i] = f"{key} = {_sanitize(val)}\n"

    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.writelines(lines)

# ==============================================
# 固定基础配置（模型、图片大小、批次等）
# ==============================================
def set_base_config():
    with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
        lines = f.readlines()
    fixed = {
        "MODEL": '"yolo12s.pt"',
        "IMGSZ": "1280",
        "BATCH": "2",
        "DEVICE": "0",
        "SINGLE_CLS": "True",
        "TIME_MINUTES": "0",
    }
    for i, line in enumerate(lines):
        for k, v in fixed.items():
            if re.match(rf"^{k}\s*=.*", line.strip()):
                lines[i] = f"{k} = {v}\n"
    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.writelines(lines)

# ==============================================
# 保存最优模型
# ==============================================
def save_best():
    os.makedirs("best_model", exist_ok=True)
    src = "autoresearch_runs/current/weights/best.pt"
    if os.path.exists(src):
        shutil.copy(src, "best_model/best.pt")

def save_final():
    os.makedirs("final_best_model", exist_ok=True)
    src = os.path.join("best_model/best.pt")
    if os.path.exists(src):
        shutil.copy(src, "final_best_model/best.pt")

# ==============================================
# 主程序
# ==============================================
def main():
    print("=" * 60)
    print("        贝叶斯超参优化 | 2 epoch/组 | 观众人头检测")
    print(f"        每组耗时 ≈ {TRAIN_EPOCHS*45} 分钟")
    print("=" * 60)

    set_base_config()
    init_results_file()  # 初始化结果文件表头
    best_score = -1
    best_m50 = best_r = best_p = 0.0
    no_improve = 0

    # 预热组（黄金参数）
    print("\n【预热组】")
    warm_up = {
        "LR0": 0.002, "LRF": 0.0002, "CONF": 0.001, "IOU": 0.5,
        "BOX": 15.0, "CLS": 1.0, "MOSAIC": 0.5, "MIXUP": 0.1,
        "COPY_PASTE": 0.1, "DEGREES": 10.0
    }
    apply_params(warm_up)
    run_training(0)
    m50, r, p = parse_results()
    score = m50 + 0.8*r + 0.4*p
    best_score = score
    best_m50, best_r, best_p = m50, r, p
    save_best()
    append_result(0, m50, r, p, score, "BEST")  # 写入结果
    print(f"预热结果 mAP50={m50:.4f} | Recall={r:.4f} | P={p:.4f}")

    # 贝叶斯循环
    for exp in range(1, MAX_EXPERIMENTS+1):
        print(f"\n======== 贝叶斯第 {exp}/{MAX_EXPERIMENTS} ========")
        print(f"当前最佳：mAP50={best_m50:.4f} Recall={best_r:.4f}")

        next_point = optimizer.ask()
        params = dict(zip([s.name for s in search_space], next_point))
        apply_params(params)

        run_training(exp)
        m50, r, p = parse_results()
        score = m50 + 0.8*r + 0.4*p
        print(f"本轮结果：mAP50={m50:.4f} Recall={r:.4f} P={p:.4f}")

        optimizer.tell(next_point, -score)
        status = "DISCARD"

        if score > best_score + 0.0005:
            best_score = score
            best_m50, best_r, best_p = m50, r, p
            no_improve = 0
            status = "BEST"
            save_best()
            print("→ 刷新最优！")
        else:
            no_improve += 1
            print(f"→ 无提升 {no_improve}/{MAX_NO_IMPROVE}")

        append_result(exp, m50, r, p, score, status)  # 写入每一轮结果

        if no_improve >= MAX_NO_IMPROVE:
            print("\n连续无提升，停止优化")
            break
        if best_m50 >= TARGET_MAP50:
            print("\n达到目标精度")
            break

    save_final()
    print("\n完成！最终最优：")
    print(f"mAP50   : {best_m50:.4f}")
    print(f"Recall  : {best_r:.4f}")
    print(f"Precision: {best_p:.4f}")

if __name__ == "__main__":
    main()
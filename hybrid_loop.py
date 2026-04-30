"""
hybrid_loop.py - 混合自主研究循环
结合贝叶斯优化的快速搜索 + Karpathy式 LLM 的有假设研究

运行:
  python hybrid_loop.py
"""

import os
import re
import sys
import json
import time
import shutil
import subprocess
import threading
import psutil
import torch
import glob
from datetime import datetime
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

from supervisor_agent import SupervisorAgent


# ==============================================
# 核心配置
# ==============================================
TRAIN_TIMEOUT       = 7200      # 2小时超时
PYTHON              = sys.executable
TRAIN_SCRIPT        = "train.py"
EVAL_SCRIPT         = "evaluate.py"
LOG_FILE            = "run.log"
RESULTS_FILE        = "results.tsv"
STATUS_FILE         = "status.md"
PROGRAM_FILE        = "program.md"
METRICS_FILE        = "last_metrics.json"
BEST_MODEL_DIR      = "best_model"

MAX_EXPERIMENTS     = 30        # 最多实验次数
BAYESIAN_PHASE_EXPS = 12        # Phase1: 贝叶斯阶段实验次数
MAX_NO_IMPROVE      = 8         # 连续无提升停止
TARGET_CDS          = 0.85      # 目标 CDS
COOLDOWN_SEC        = 30        # 冷却时间

# LLM 配置
OLLAMA_MODEL        = "gemma3:4b"
OLLAMA_URL          = "http://localhost:11434"

# 硬件安全
MAX_VRAM_USAGE      = 0.90
MAX_RAM_USAGE       = 0.85
MIN_FREE_DISK_GB    = 10


# ==============================================
# 贝叶斯搜索空间
# ==============================================
SEARCH_SPACE = [
    Real(0.001, 0.02, name="LR0"),
    Real(0.0005, 0.02, name="LRF"),
    Real(0.0005, 0.002, name="CONF"),
    Real(0.4, 0.7, name="IOU"),
    Real(5.0, 20.0, name="BOX"),
    Real(0.3, 1.5, name="CLS"),
    Real(0.5, 1.0, name="MOSAIC"),
    Real(0.0, 0.3, name="MIXUP"),
    Real(0.0, 0.3, name="COPY_PASTE"),
    Real(0.0, 20.0, name="DEGREES"),
]


# ==============================================
# 工具函数
# ==============================================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def init_files():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("exp\tdate\tphase\tdescription\tCDS\tmAP50\tmAP50-95\tprecision\trecall\tsmall_obj_recall\tcounting_mae\tlatency_ms\tstatus\n")
    if not os.path.exists(BEST_MODEL_DIR):
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ==============================================
# 硬件安全
# ==============================================
def check_hardware_safety():
    try:
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_used = torch.cuda.memory_allocated(0)
            vram_ratio = vram_used / vram_total
            if vram_ratio > MAX_VRAM_USAGE:
                log(f"⚠️ VRAM 高: {vram_ratio:.1%}")
                torch.cuda.empty_cache()
                return False
        
        ram = psutil.virtual_memory()
        ram_used = ram.used / ram.total
        if ram_used > MAX_RAM_USAGE:
            log(f"⚠️ RAM 高: {ram_used:.1%}")
            return False
        
        disk = psutil.disk_usage(".")
        free_gb = disk.free / (1024**3)
        if free_gb < MIN_FREE_DISK_GB:
            log(f"⚠️ 磁盘不足: {free_gb:.1f} GB")
            return False
        
        return True
    except Exception as e:
        log(f"⚠️ 硬件检查: {e}")
        return True


def cooldown():
    log(f"⏱️ 冷却 {COOLDOWN_SEC} 秒...")
    time.sleep(COOLDOWN_SEC)


# ==============================================
# Git 工作流
# ==============================================
def git_command(cmd):
    try:
        result = subprocess.run(
            ["git"] + cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def git_commit(description):
    try:
        git_command(["add", TRAIN_SCRIPT])
        commit_msg = f"autoresearch: {description}"
        code, _, err = git_command(["commit", "-m", commit_msg])
        if code == 0:
            log(f"✅ Git commit: {description}")
            return True
        else:
            log(f"⚠️ Git commit: {err}")
            return False
    except Exception as e:
        log(f"⚠️ Git commit 失败: {e}")
        return False


def git_reset():
    try:
        code, _, err = git_command(["reset", "--hard", "HEAD~1"])
        if code == 0:
            log(f"✅ Git reset")
            return True
        else:
            log(f"⚠️ Git reset: {err}")
            return False
    except Exception as e:
        log(f"⚠️ Git reset 失败: {e}")
        return False


def git_push():
    try:
        log("📤 推送到远程...")
        code, _, err = git_command(["push"])
        if code == 0:
            log("✅ 推送成功")
            return True
        else:
            log(f"⚠️ 推送: {err}")
            return False
    except Exception as e:
        log(f"⚠️ 推送失败: {e}")
        return False


# ==============================================
# 训练与评估
# ==============================================
def run_training(exp_num):
    log(f"🚀 实验 #{exp_num} 训练...")
    
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    try:
        proc = subprocess.Popen(
            [PYTHON, "-u", TRAIN_SCRIPT],
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
                log("⚠️ 训练超时")
            except:
                pass
        
        threading.Thread(target=watchdog, daemon=True).start()
        
        # 实时输出 — flush=True 确保 Windows cmd/PowerShell 立刻显示。
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                print(line.rstrip(), flush=True)
        
        proc.wait()
        return proc.returncode == 0
        
    except Exception as e:
        log(f"⚠️ 训练出错: {e}")
        return False


def run_evaluation():
    log("🔍 评估...")
    try:
        subprocess.run([PYTHON, "-u", EVAL_SCRIPT], capture_output=False, check=True)
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            log(f"✅ 评估完成: CDS={metrics.get('CDS', 0):.4f}")
            return metrics
        return None
    except Exception as e:
        log(f"⚠️ 评估出错: {e}")
        return None


# ==============================================
# train.py 操作
# ==============================================
def read_train_py():
    try:
        with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def apply_bayesian_params(params_dict):
    """应用贝叶斯参数到 train.py"""
    try:
        content = read_train_py()
        for key, value in params_dict.items():
            # 匹配 "KEY = value" 格式
            pattern = rf"^{key}\s*=.*$"
            replacement = f"{key} = {value}"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
            f.write(content)
        
        log(f"✅ 应用贝叶斯参数: {params_dict}")
        return True
        
    except Exception as e:
        log(f"⚠️ 应用参数失败: {e}")
        return False


def apply_llm_train_py(new_content):
    try:
        with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
            f.write(new_content)
        log("✅ LLM 修改已应用")
        return True
    except Exception as e:
        log(f"⚠️ 应用失败: {e}")
        return False


# ==============================================
# LLM 集成
# ==============================================
def call_ollama(prompt):
    try:
        import httpx
    except ImportError:
        log("⚠️ 需要: pip install httpx")
        return None
    
    try:
        client = httpx.Client(timeout=120.0)
        response = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "num_predict": 2048
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            log(f"⚠️ Ollama 错误: {response.status_code}")
            return None
    except Exception as e:
        log(f"⚠️ Ollama 连接失败: {e}")
        return None


def read_program():
    try:
        with open(PROGRAM_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def read_results_history(n=5):
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= 1:
            return "暂无历史"
        header = lines[0]
        recent = lines[-n:] if len(lines) > n + 1 else lines[1:]
        return header + "".join(recent)
    except:
        return ""


def llm_propose_with_bayesian_hint(exp_num, best_cds, history, bayesian_hint):
    """LLM 结合贝叶斯提示提出假设"""
    program = read_program()
    current_train = read_train_py()
    
    prompt = f"""你是一个计算机视觉研究专家，正在做体育场密集人群检测的研究。

研究规则:
{program}

当前 train.py:
{current_train}

当前最佳 CDS: {best_cds:.4f}

最近实验历史:
{history}

贝叶斯优化推荐的参数范围（作为参考）:
{json.dumps(bayesian_hint, indent=2)}

现在请为实验 #{exp_num} 提出一个假设。你可以：
1. 参考贝叶斯推荐的参数范围
2. 或者尝试其他有创意的想法（数据增强策略、损失权重调整等）

请按格式输出：

DESCRIPTION: [一句话描述假设]

[然后输出完整的修改后的 train.py]
"""
    
    log("🧠 LLM 正在思考（参考贝叶斯提示）...")
    response = call_ollama(prompt)
    
    if not response:
        return None, None
    
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", response)
    if not desc_match:
        log("⚠️ 无法解析 LLM 输出")
        return None, None
    
    description = desc_match.group(1).strip()
    
    code_start = response.find("```python")
    code_end = response.find("```", code_start + 1) if code_start != -1 else -1
    
    if code_start != -1 and code_end != -1:
        new_train = response[code_start + 9: code_end].strip()
    else:
        model_pos = response.find("MODEL = ")
        if model_pos != -1:
            new_train = response[model_pos:].strip()
        else:
            log("⚠️ 无法找到 train.py 代码")
            return None, None
    
    return description, new_train


# ==============================================
# 结果记录
# ==============================================
def append_result(exp_num, phase, description, metrics, status):
    try:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cds = metrics.get("CDS", 0)
        mAP50 = metrics.get("mAP50", 0)
        mAP50_95 = metrics.get("mAP50-95", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        small_obj_recall = metrics.get("small_obj_recall", 0)
        counting_mae = metrics.get("counting_mae", 0)
        latency_ms = metrics.get("inference_ms", 0)
        
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{exp_num}\t{date}\t{phase}\t{description}\t{cds:.4f}\t{mAP50:.4f}\t{mAP50_95:.4f}\t{precision:.4f}\t{recall:.4f}\t{small_obj_recall:.4f}\t{counting_mae:.4f}\t{latency_ms:.1f}\t{status}\n")
        
        log(f"📝 记录: {status}")
    except Exception as e:
        log(f"⚠️ 记录失败: {e}")


def update_status(exp_num, phase, best_cds, best_metrics, no_improve):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(f"""# Hybrid AutoResearch Status

## 当前状态
- **阶段**: {phase}
- **实验**: {exp_num} / {MAX_EXPERIMENTS}
- **连续无提升**: {no_improve} / {MAX_NO_IMPROVE}
- **最佳 CDS**: {best_cds:.4f}

## 最佳指标
- mAP50: {best_metrics.get('mAP50', 0):.4f}
- mAP50-95: {best_metrics.get('mAP50-95', 0):.4f}
- Precision: {best_metrics.get('precision', 0):.4f}
- Recall: {best_metrics.get('recall', 0):.4f}
- Small Object Recall: {best_metrics.get('small_obj_recall', 0):.4f}
- Counting MAE: {best_metrics.get('counting_mae', 0):.2f}
- Latency: {best_metrics.get('inference_ms', 0):.1f} ms

## Quality Gates
- Precision >= 0.90: {'✅' if best_metrics.get('precision', 0) >= 0.90 else '❌'}
- Recall >= 0.85: {'✅' if best_metrics.get('recall', 0) >= 0.85 else '❌'}

---

历史: `cat {RESULTS_FILE}`
""")
    except Exception as e:
        log(f"⚠️ 更新 status.md 失败: {e}")


def save_best_model():
    try:
        src_pattern = "autoresearch_runs/*/weights/best.pt"
        candidates = glob.glob(src_pattern)
        if candidates:
            candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            src = candidates[0]
            dst = os.path.join(BEST_MODEL_DIR, "best.pt")
            shutil.copy2(src, dst)
            log(f"💾 最佳模型已保存")
    except Exception as e:
        log(f"⚠️ 保存模型失败: {e}")


# ==============================================
# 主循环
# ==============================================
def main():
    print("=" * 80)
    print("        HYBRID AUTORESEARCH LOOP")
    print("=" * 80)
    
    init_files()
    
    supervisor = SupervisorAgent()
    supervisor.load_existing_results()
    
    optimizer = Optimizer(dimensions=SEARCH_SPACE, base_estimator="GP", random_state=42)
    
    best_cds = supervisor.best_cds if supervisor.best_cds > 0 else -1.0
    best_metrics = supervisor.best_metrics if supervisor.best_metrics else {}
    no_improve = supervisor.consecutive_no_improve
    exp_num = supervisor.total_experiments + 1
    
    log(f"📋 总实验: {MAX_EXPERIMENTS}")
    log(f"🧮 贝叶斯阶段: {BAYESIAN_PHASE_EXPS} 次")
    log(f"🧠 LLM 阶段: {MAX_EXPERIMENTS - BAYESIAN_PHASE_EXPS} 次")
    log(f"🎯 目标 CDS: {TARGET_CDS}")
    
    while exp_num <= MAX_EXPERIMENTS:
        print()
        log("=" * 60)
        
        # 确定阶段
        if exp_num <= BAYESIAN_PHASE_EXPS:
            phase = "BAYESIAN"
            log(f"       实验 {exp_num}/{MAX_EXPERIMENTS} | 阶段: 📊 {phase}")
        else:
            phase = "LLM"
            log(f"       实验 {exp_num}/{MAX_EXPERIMENTS} | 阶段: 🧠 {phase}")
        
        log("=" * 60)
        
        # 硬件检查
        if not check_hardware_safety():
            cooldown()
        
        # 读取历史
        history = read_results_history(5)
        
        # 提出假设
        if phase == "BAYESIAN":
            # Phase 1: 贝叶斯优化
            next_point = optimizer.ask()
            param_names = [dim.name for dim in SEARCH_SPACE]
            params_dict = dict(zip(param_names, next_point))
            description = f"Bayesian opt: {json.dumps(params_dict, separators=(',', ':'))}"
            log(f"📊 贝叶斯推荐: {params_dict}")
            success = apply_bayesian_params(params_dict)
        else:
            # Phase 2: LLM + 贝叶斯提示
            # 获取贝叶斯对当前最佳的建议
            try:
                bayesian_hint = {}
                for dim in SEARCH_SPACE:
                    # 简单提示：给出当前搜索空间的最佳位置
                    samples = optimizer.ask(n_points=5)
                    param_name = dim.name
                    values = [s[i] for i, s in enumerate(param_names) if param_names[i] == param_name]
                    if values:
                        bayesian_hint[param_name] = {
                            "min": min(values),
                            "max": max(values),
                            "suggested": sum(values) / len(values)
                        }
            except:
                bayesian_hint = {}
            
            description, new_train_py = llm_propose_with_bayesian_hint(
                exp_num, best_cds, history, bayesian_hint
            )
            
            if not description or not new_train_py:
                log("⚠️ LLM 无法提出假设，跳过")
                exp_num += 1
                cooldown()
                continue
            
            log(f"💡 假设: {description}")
            success = apply_llm_train_py(new_train_py)
        
        if not success:
            exp_num += 1
            cooldown()
            continue
        
        # Git commit
        git_commit(f"[{phase}] {description}")
        
        # 训练
        train_ok = run_training(exp_num)
        
        if not train_ok:
            log("❌ 训练失败")
            append_result(exp_num, phase, description, {"CDS": 0}, "CRASH")
            git_reset()
            exp_num += 1
            cooldown()
            continue
        
        # 评估
        metrics = run_evaluation()
        
        if not metrics:
            log("❌ 评估失败")
            append_result(exp_num, phase, description, {"CDS": 0}, "EVAL_FAIL")
            git_reset()
            exp_num += 1
            cooldown()
            continue
        
        # 贝叶斯更新（即使是 LLM 阶段也更新贝叶斯）
        cds = metrics.get("CDS", 0)
        
        if phase == "BAYESIAN":
            optimizer.tell(next_point, -cds)
        
        # 决策
        log(f"📊 CDS: {cds:.4f} | Best: {best_cds:.4f}")
        
        if cds > best_cds + 0.001:
            log("🎉 有提升！KEEP")
            best_cds = cds
            best_metrics = metrics.copy()
            no_improve = 0
            status = "KEEP"
            save_best_model()
            git_push()
        else:
            log("📉 无提升，DISCARD")
            no_improve += 1
            status = "DISCARD"
            git_reset()
        
        supervisor.audit_new_result(metrics, status)
        
        safety_issues = supervisor.safety_guard.check_hardware()
        for level, msg in safety_issues:
            if level == "CRIT":
                log(f"🚨 监督告警: {msg}")
        
        # 记录
        append_result(exp_num, phase, description, metrics, status)
        update_status(exp_num, phase, best_cds, best_metrics, no_improve)
        
        # 停止条件
        if no_improve >= MAX_NO_IMPROVE:
            log()
            log("=" * 60)
            log(f"🛑 连续 {MAX_NO_IMPROVE} 次无提升，停止")
            log("=" * 60)
            break
        
        if best_cds >= TARGET_CDS:
            log()
            log("=" * 60)
            log(f"🏆 达到目标 CDS={TARGET_CDS}！")
            log("=" * 60)
            break
        
        if (best_metrics.get("precision", 0) >= 0.90 and
            best_metrics.get("recall", 0) >= 0.85):
            log()
            log("=" * 60)
            log("🎯 Quality Gates 全部达标！")
            log("=" * 60)
        
        exp_num += 1
        cooldown()
    
    # 结束
    print()
    log("=" * 60)
    log("        研究完成！")
    log("=" * 60)
    log(f"最佳 CDS: {best_cds:.4f}")
    log(f"最佳指标: {json.dumps(best_metrics, indent=2)}")
    log(f"最佳模型: {BEST_MODEL_DIR}/best.pt")
    log(f"历史: {RESULTS_FILE}")
    log("=" * 60)
    
    supervisor.report_generator.generate_report()


if __name__ == "__main__":
    main()

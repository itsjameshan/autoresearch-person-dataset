"""
autoresearch_loop.py - Karpathy式自主研究循环
基于 Andrej Karpathy 的 autoresearch 理念
LLM 扮演 PhD 学生，自主运行研究循环

运行:
  1. 确保 Ollama 运行:
     (Windows) $env:OLLAMA_GPU_LAYERS = 0; ollama serve
  2. 运行: python autoresearch_loop.py
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
from datetime import datetime


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
MAX_NO_IMPROVE      = 8         # 连续无提升停止
TARGET_CDS          = 0.85      # 目标CDS
COOLDOWN_SEC        = 30        # 冷却时间（30秒）

# LLM 配置
OLLAMA_MODEL        = "gemma3:4b"  # 或 "llama3.1:8b"
OLLAMA_URL          = "http://localhost:11434"

# 硬件安全控制
MAX_VRAM_USAGE      = 0.90      # VRAM 使用率上限
MAX_RAM_USAGE       = 0.85      # RAM 使用率上限
MIN_FREE_DISK_GB    = 10        # 最小剩余磁盘空间

# ==============================================
# 工具函数
# ==============================================
def log(msg: str):
    """带时间戳的日志"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def init_files():
    """初始化必要的文件"""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("exp\tdate\tdescription\tCDS\tmAP50\tmAP50-95\tprecision\trecall\tsmall_obj_recall\tcounting_mae\tlatency_ms\tstatus\n")
    
    if not os.path.exists(BEST_MODEL_DIR):
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ==============================================
# 硬件安全检查
# ==============================================
def check_hardware_safety():
    """检查硬件是否安全，返回 True=安全, False=不安全"""
    try:
        # 1. 检查 GPU VRAM
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_used = torch.cuda.memory_allocated(0)
            vram_ratio = vram_used / vram_total
            if vram_ratio > MAX_VRAM_USAGE:
                log(f"⚠️ VRAM 使用率过高: {vram_ratio:.1%}")
                torch.cuda.empty_cache()
                return False
        
        # 2. 检查 RAM
        ram = psutil.virtual_memory()
        ram_used = ram.used / ram.total
        if ram_used > MAX_RAM_USAGE:
            log(f"⚠️ RAM 使用率过高: {ram_used:.1%}")
            return False
        
        # 3. 检查磁盘空间
        disk = psutil.disk_usage(".")
        free_gb = disk.free / (1024**3)
        if free_gb < MIN_FREE_DISK_GB:
            log(f"⚠️ 磁盘空间不足: {free_gb:.1f} GB 剩余")
            return False
        
        return True
    except Exception as e:
        log(f"⚠️ 硬件检查出错: {e}")
        return True  # 放宽，继续运行


def cooldown():
    """硬件冷却等待"""
    log(f"⏱️ 冷却等待 {COOLDOWN_SEC} 秒...")
    time.sleep(COOLDOWN_SEC)


# ==============================================
# Git 工作流
# ==============================================
def git_command(cmd: list[str]) -> tuple[int, str, str]:
    """运行 Git 命令"""
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


def git_commit(description: str) -> bool:
    """提交当前修改"""
    try:
        git_command(["add", TRAIN_SCRIPT])
        commit_msg = f"autoresearch: {description}"
        code, _, err = git_command(["commit", "-m", commit_msg])
        if code == 0:
            log(f"✅ Git 提交成功: {description}")
            return True
        else:
            log(f"⚠️ Git 提交: {err}")
            return False
    except Exception as e:
        log(f"⚠️ Git 提交失败: {e}")
        return False


def git_reset() -> bool:
    """重置到上一次提交"""
    try:
        code, _, err = git_command(["reset", "--hard", "HEAD~1"])
        if code == 0:
            log(f"✅ Git 重置成功")
            return True
        else:
            log(f"⚠️ Git 重置: {err}")
            return False
    except Exception as e:
        log(f"⚠️ Git 重置失败: {e}")
        return False


def git_push() -> bool:
    """推送到远程（可选）"""
    try:
        log("📤 推送到远程仓库...")
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
def run_training(exp_num: int) -> bool:
    """运行训练，返回是否成功"""
    log(f"🚀 开始实验 #{exp_num} 训练...")
    
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    
    # 清空旧日志
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    # 启动训练
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
        
        # 超时监控
        def watchdog():
            time.sleep(TRAIN_TIMEOUT)
            try:
                proc.kill()
                log("⚠️ 训练超时被终止")
            except:
                pass
        
        threading.Thread(target=watchdog, daemon=True).start()
        
        # 实时输出
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                print(line.rstrip())
        
        proc.wait()
        return proc.returncode == 0
        
    except Exception as e:
        log(f"⚠️ 训练出错: {e}")
        return False


def run_evaluation() -> dict | None:
    """运行评估，返回指标字典"""
    log("🔍 开始评估...")
    
    try:
        # 运行 evaluate.py
        subprocess.run(
            [PYTHON, "-u", EVAL_SCRIPT],
            capture_output=False,
            check=True
        )
        
        # 读取指标
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
# LLM 集成（使用 Ollama）
# ==============================================
def call_ollama(prompt: str) -> str:
    """调用 Ollama"""
    try:
        import httpx
    except ImportError:
        log("⚠️ 需要安装 httpx: pip install httpx")
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
            result = response.json()
            return result.get("response", "")
        else:
            log(f"⚠️ Ollama 调用失败: {response.status_code}")
            return None
            
    except Exception as e:
        log(f"⚠️ Ollama 连接失败: {e}")
        log("提示: 确保 Ollama 正在运行")
        return None


def read_program() -> str:
    """读取 program.md"""
    try:
        with open(PROGRAM_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def read_train_py() -> str:
    """读取 train.py 当前内容"""
    try:
        with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def read_results_history(n: int = 5) -> str:
    """读取最近 n 条结果"""
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= 1:
            return "暂无历史记录"
        
        header = lines[0]
        recent = lines[-n:] if len(lines) > n + 1 else lines[1:]
        return header + "".join(recent)
    except:
        return ""


def llm_propose_hypothesis(exp_num: int, best_cds: float, history: str) -> tuple[str, str]:
    """
    LLM 提出假设
    返回: (description, 修改后的 train.py)
    """
    program = read_program()
    current_train = read_train_py()
    
    prompt = f"""你是一个计算机视觉研究专家，正在做体育场密集人群检测的研究。

请阅读研究规则：
{program}

当前 train.py 内容:
{current_train}

当前最佳 CDS: {best_cds:.4f}

最近实验历史:
{history}

现在请为实验 #{exp_num} 提出一个假设。

请按以下格式输出：

DESCRIPTION: [一句话描述你的假设和为什么]

[然后直接输出完整修改后的 train.py 内容]
"""
    
    log("🧠 请 LLM 提出假设...")
    response = call_ollama(prompt)
    
    if not response:
        return None, None
    
    # 解析 DESCRIPTION 和 train.py
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", response)
    if not desc_match:
        log("⚠️ 无法解析 LLM 输出")
        return None, None
    
    description = desc_match.group(1).strip()
    
    # 尝试提取 train.py 代码
    code_start = response.find("```python")
    code_end = response.find("```", code_start + 1) if code_start != -1 else -1
    
    if code_start != -1 and code_end != -1:
        new_train_py = response[code_start + 9 : code_end].strip()
    else:
        # 尝试找 "MODEL = " 作为起点
        model_pos = response.find("MODEL = ")
        if model_pos != -1:
            new_train_py = response[model_pos:].strip()
        else:
            log("⚠️ 无法找到 train.py 代码")
            return None, None
    
    return description, new_train_py


# ==============================================
# 修改 train.py
# ==============================================
def apply_train_py(new_content: str) -> bool:
    """用 LLM 输出的内容覆盖 train.py"""
    try:
        with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
            f.write(new_content)
        log("✅ train.py 已更新")
        return True
    except Exception as e:
        log(f"⚠️ 更新 train.py 失败: {e}")
        return False


# ==============================================
# 结果记录
# ==============================================
def append_result(exp_num: int, description: str, metrics: dict, status: str):
    """记录实验结果"""
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
            f.write(f"{exp_num}\t{date}\t{description}\t{cds:.4f}\t{mAP50:.4f}\t{mAP50_95:.4f}\t{precision:.4f}\t{recall:.4f}\t{small_obj_recall:.4f}\t{counting_mae:.4f}\t{latency_ms:.1f}\t{status}\n")
        
        log(f"📝 结果已记录: {status}")
        
    except Exception as e:
        log(f"⚠️ 记录结果失败: {e}")


def update_status(exp_num: int, best_cds: float, best_metrics: dict, no_improve: int):
    """更新 status.md"""
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(f"""# AutoResearch Status

## 当前状态
- **实验进度**: {exp_num} / {MAX_EXPERIMENTS}
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

查看完整历史: `cat {RESULTS_FILE}`
""")
    except Exception as e:
        log(f"⚠️ 更新 status.md 失败: {e}")


def save_best_model():
    """保存最佳模型"""
    try:
        # 从 autoresearch_runs 找最新的 best.pt
        src_pattern = "autoresearch_runs/*/weights/best.pt"
        candidates = glob.glob(src_pattern)
        if candidates:
            # 选最新修改的
            candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            src = candidates[0]
            dst = os.path.join(BEST_MODEL_DIR, "best.pt")
            shutil.copy2(src, dst)
            log(f"💾 最佳模型已保存: {dst}")
    except Exception as e:
        log(f"⚠️ 保存最佳模型失败: {e}")


# ==============================================
# 主循环
# ==============================================
def main():
    import glob
    
    print("=" * 80)
    print("        KARPATHY-STYLE AUTORESEARCH LOOP")
    print("=" * 80)
    
    init_files()
    
    best_cds = -1.0
    best_metrics = {}
    no_improve = 0
    exp_num = 1
    
    log(f"📋 最大实验次数: {MAX_EXPERIMENTS}")
    log(f"🎯 目标 CDS: {TARGET_CDS}")
    log(f"🛑 连续无提升停止: {MAX_NO_IMPROVE}")
    
    # 预检查
    if not check_hardware_safety():
        log("⚠️ 硬件状态不佳，但继续尝试...")
    
    while exp_num <= MAX_EXPERIMENTS:
        print()
        log("=" * 60)
        log(f"       实验 {exp_num} / {MAX_EXPERIMENTS}")
        log("=" * 60)
        
        # 1. 硬件检查
        if not check_hardware_safety():
            cooldown()
        
        # 2. 读取历史
        history = read_results_history(5)
        
        # 3. LLM 提出假设
        description, new_train_py = llm_propose_hypothesis(
            exp_num, best_cds, history
        )
        
        if not description or not new_train_py:
            log("⚠️ LLM 无法提出假设，跳过本次")
            exp_num += 1
            cooldown()
            continue
        
        log(f"💡 假设: {description}")
        
        # 4. 应用修改
        if not apply_train_py(new_train_py):
            exp_num += 1
            cooldown()
            continue
        
        # 5. Git 提交
        git_commit(description)
        
        # 6. 训练
        success = run_training(exp_num)
        
        if not success:
            log("❌ 训练失败")
            append_result(exp_num, description, {"CDS": 0}, "CRASH")
            git_reset()
            exp_num += 1
            cooldown()
            continue
        
        # 7. 评估
        metrics = run_evaluation()
        
        if not metrics:
            log("❌ 评估失败")
            append_result(exp_num, description, {"CDS": 0}, "EVAL_FAIL")
            git_reset()
            exp_num += 1
            cooldown()
            continue
        
        # 8. 决策
        cds = metrics.get("CDS", 0)
        log(f"📊 当前 CDS: {cds:.4f} | 最佳 CDS: {best_cds:.4f}")
        
        if cds > best_cds + 0.001:
            # 有提升，保留
            log("🎉 有提升！KEEP")
            best_cds = cds
            best_metrics = metrics.copy()
            no_improve = 0
            status = "KEEP"
            save_best_model()
            git_push()  # 可选推送
        else:
            # 无提升，丢弃
            log("📉 无提升，DISCARD")
            no_improve += 1
            status = "DISCARD"
            git_reset()
        
        # 9. 记录
        append_result(exp_num, description, metrics, status)
        update_status(exp_num, best_cds, best_metrics, no_improve)
        
        # 10. 检查停止条件
        if no_improve >= MAX_NO_IMPROVE:
            log()
            log("=" * 60)
            log(f"🛑 连续 {MAX_NO_IMPROVE} 次无提升，停止研究")
            log("=" * 60)
            break
        
        if best_cds >= TARGET_CDS:
            log()
            log("=" * 60)
            log(f"🏆 达到目标 CDS={TARGET_CDS}！停止研究")
            log("=" * 60)
            break
        
        if (best_metrics.get("precision", 0) >= 0.90 and
            best_metrics.get("recall", 0) >= 0.85):
            log()
            log("=" * 60)
            log("🎯 Quality Gates 全部达标！")
            log("=" * 60)
        
        # 11. 冷却
        exp_num += 1
        cooldown()
    
    # 结束
    print()
    log("=" * 60)
    log("        研究完成！")
    log("=" * 60)
    log(f"最终最佳 CDS: {best_cds:.4f}")
    log(f"最终指标: {json.dumps(best_metrics, indent=2)}")
    log(f"最佳模型已保存至: {BEST_MODEL_DIR}/best.pt")
    log(f"完整历史: {RESULTS_FILE}")
    log(f"状态概览: {STATUS_FILE}")
    log("=" * 60)


if __name__ == "__main__":
    main()

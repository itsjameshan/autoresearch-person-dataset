"""
supervisor_dashboard.py — 智能体总监视面板 v2
多数据源实时监控 Bilevel 自主优化训练进度
"""

import os
import json
import time
import re
import glob
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "bilevel_run.log")
CONSOLE_LOG = os.path.join(BASE_DIR, "bilevel_console.log")
RESULTS_TSV = os.path.join(BASE_DIR, "bilevel_results.tsv")

TRAIN_ROOT = os.path.join(BASE_DIR, "runs", "detect", "autoresearch_runs")

app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bilevel Autoresearch — 智能体总监视面板</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            font-size: 24px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            background: #10b981;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }
        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
        }
        .card-label {
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 28px;
            font-weight: 700;
            color: #f1f5f9;
        }
        .card-sub {
            font-size: 12px;
            color: #64748b;
            margin-top: 4px;
        }
        .progress-section {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
            margin-bottom: 20px;
        }
        .progress-header {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #94a3b8;
            margin-bottom: 12px;
        }
        .progress-bar {
            height: 12px;
            background: #334155;
            border-radius: 999px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #7c3aed, #06b6d4);
            border-radius: 999px;
            transition: width 0.5s ease;
        }
        .two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #f1f5f9;
        }
        .log-section {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 12px;
            line-height: 1.6;
            max-height: 400px;
            overflow-y: auto;
            color: #94a3b8;
            white-space: pre-wrap;
            word-break: break-all;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }
        th {
            color: #94a3b8;
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        tr:hover { background: #1e293b; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-best { background: #064e3b; color: #34d399; }
        .badge-sa { background: #78350f; color: #fbbf24; }
        .badge-discard { background: #450a0a; color: #f87171; }
        .badge-crash { background: #450a0a; color: #f87171; }
        .badge-running { background: #1e3a5f; color: #60a5fa; }
        .params-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            font-size: 12px;
        }
        .param-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid #1e293b;
        }
        .param-name { color: #94a3b8; }
        .param-value { color: #e2e8f0; font-family: monospace; }
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 12px;
        }
        .metric-mini {
            background: #0f172a;
            border-radius: 6px;
            padding: 8px;
            text-align: center;
        }
        .metric-mini-label {
            font-size: 11px;
            color: #64748b;
        }
        .metric-mini-value {
            font-size: 14px;
            font-weight: 600;
            color: #a78bfa;
            font-family: monospace;
        }
        .refresh-info {
            text-align: center;
            color: #64748b;
            font-size: 12px;
            margin-top: 12px;
        }
        .source-tag {
            display: inline-block;
            padding: 1px 6px;
            background: #334155;
            color: #94a3b8;
            border-radius: 4px;
            font-size: 10px;
            margin-left: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <span class="status-dot" id="status-dot"></span>
            Bilevel Autoresearch — 智能体总监视面板
        </h1>

        <div class="grid">
            <div class="card">
                <div class="card-label">当前阶段</div>
                <div class="card-value" id="phase">加载中...</div>
                <div class="card-sub" id="phase-sub"></div>
            </div>
            <div class="card">
                <div class="card-label">实验数</div>
                <div class="card-value" id="exp-count">0</div>
                <div class="card-sub">目标: 100</div>
            </div>
            <div class="card">
                <div class="card-label">最佳 CDS</div>
                <div class="card-value" id="best-cds">—</div>
                <div class="card-sub" id="baseline-cds">基线: —</div>
            </div>
            <div class="card">
                <div class="card-label">优化引擎</div>
                <div class="card-value" id="engine" style="font-size:18px;">LLM</div>
                <div class="card-sub" id="model-info">模型: qwen2.5-coder:7b</div>
            </div>
        </div>

        <div class="progress-section">
            <div class="progress-header">
                <span>当前实验进度 <span class="source-tag" id="data-source">数据源: —</span></span>
                <span id="epoch-info">等待中...</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
            <div class="metrics-row" id="train-metrics">
            </div>
        </div>

        <div class="two-col">
            <div class="card" style="border: 1px solid #334155;">
                <div class="section-title">📊 实验历史</div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>外循环</th>
                                <th>CDS</th>
                                <th>状态</th>
                            </tr>
                        </thead>
                        <tbody id="experiments-body">
                            <tr><td colspan="4" style="text-align:center;color:#64748b;">等待第一个实验完成...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card" style="border: 1px solid #334155;">
                <div class="section-title">⚙️ 当前最佳参数</div>
                <div class="params-grid" id="best-params">
                    <div class="param-row"><span class="param-name">LR0</span><span class="param-value">0.002</span></div>
                    <div class="param-row"><span class="param-name">MOSAIC</span><span class="param-value">1.0</span></div>
                    <div class="param-row"><span class="param-name">BATCH</span><span class="param-value">4</span></div>
                    <div class="param-row"><span class="param-name">IMGSZ</span><span class="param-value">960</span></div>
                </div>
            </div>
        </div>

        <div class="card" style="border: 1px solid #334155; margin-bottom: 20px;">
            <div class="section-title">📝 实时训练日志</div>
            <div class="log-section" id="log">加载中...</div>
        </div>

        <div class="refresh-info">
            <span id="last-update">最后更新: —</span>
            <span style="margin: 0 8px;">|</span>
            <span>自动刷新: 每 3 秒</span>
        </div>
    </div>

    <script>
        async function updateDashboard() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                document.getElementById('phase').textContent = data.phase;
                document.getElementById('phase-sub').textContent = data.phase_sub;
                document.getElementById('exp-count').textContent = data.exp_count;
                document.getElementById('best-cds').textContent = data.best_cds;
                document.getElementById('baseline-cds').textContent = '基线: ' + data.baseline_cds;
                document.getElementById('epoch-info').textContent = data.epoch_info;
                document.getElementById('progress-fill').style.width = data.progress + '%';
                document.getElementById('data-source').textContent = '数据源: ' + data.data_source;
                document.getElementById('log').textContent = data.log_tail;

                const logEl = document.getElementById('log');
                logEl.scrollTop = logEl.scrollHeight;

                const tbody = document.getElementById('experiments-body');
                if (data.experiments && data.experiments.length > 0) {
                    tbody.innerHTML = data.experiments.map(e => `
                        <tr>
                            <td>${e.id}</td>
                            <td>${e.outer}</td>
                            <td style="font-family:monospace;">${e.cds}</td>
                            <td><span class="badge badge-${e.status_class}">${e.status}</span></td>
                        </tr>
                    `).join('');
                }

                const metricsDiv = document.getElementById('train-metrics');
                if (data.train_metrics && Object.keys(data.train_metrics).length > 0) {
                    metricsDiv.innerHTML = Object.entries(data.train_metrics).map(([k, v]) => `
                        <div class="metric-mini">
                            <div class="metric-mini-label">${k}</div>
                            <div class="metric-mini-value">${v}</div>
                        </div>
                    `).join('');
                } else {
                    metricsDiv.innerHTML = '';
                }

                document.getElementById('last-update').textContent = '最后更新: ' + data.timestamp;

                const dot = document.getElementById('status-dot');
                dot.style.background = data.running ? '#10b981' : '#f59e0b';
            } catch (e) {
                console.error('Update failed:', e);
            }
        }

        updateDashboard();
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>
"""


def get_latest_train_dir():
    """获取最新的训练目录"""
    if not os.path.exists(TRAIN_ROOT):
        return None
    dirs = [d for d in glob.glob(os.path.join(TRAIN_ROOT, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return dirs[0]


def read_log_tail(filepath, lines=50):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            if not all_lines:
                return None
            return "".join(all_lines[-lines:])
    except:
        return None


def parse_results_tsv():
    """解析 bilevel_results.tsv 实验历史"""
    experiments = []
    if not os.path.exists(RESULTS_TSV):
        return experiments
    try:
        with open(RESULTS_TSV, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[1:], 1):
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    status = parts[4]
                    status_class = "best" if status == "BEST" else \
                                   "sa" if status == "SA_ACCEPT" else \
                                   "discard" if status == "DISCARD" else \
                                   "crash" if status == "CRASH" else "running"
                    experiments.append({
                        "id": i,
                        "outer": parts[0] if len(parts) > 0 else "-",
                        "cds": parts[3] if len(parts) > 3 else "-",
                        "status": status,
                        "status_class": status_class
                    })
        return list(reversed(experiments))[:20]
    except:
        return experiments


def parse_results_csv(train_dir):
    """从 Ultralytics results.csv 解析训练进度和指标"""
    csv_path = os.path.join(train_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None, 0, {}
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None, 0, {}
            
            header = lines[0].strip().split(",")
            last_line = lines[-1].strip().split(",")
            
            epoch_idx = None
            for i, h in enumerate(header):
                if "epoch" in h.lower():
                    epoch_idx = i
                    break
            
            if epoch_idx is None or epoch_idx >= len(last_line):
                return None, 0, {}
            
            cur_epoch = int(float(last_line[epoch_idx].strip()))
            
            train_metrics = {}
            metric_map = {
                "train/box_loss": "Box Loss",
                "train/cls_loss": "Cls Loss", 
                "train/dfl_loss": "DFL Loss",
                "metrics/precision(B)": "Precision",
                "metrics/recall(B)": "Recall",
                "metrics/mAP50(B)": "mAP50",
                "metrics/mAP50-95(B)": "mAP50-95",
                "val/box_loss": "Val Box",
            }
            
            for csv_name, display_name in metric_map.items():
                for i, h in enumerate(header):
                    if csv_name in h.strip() and i < len(last_line):
                        try:
                            val = float(last_line[i].strip())
                            train_metrics[display_name] = f"{val:.4f}"
                        except:
                            pass
            
            return f"Epoch {cur_epoch} (从 results.csv)", (cur_epoch / 32) * 100, train_metrics
    except:
        return None, 0, {}


def parse_epoch_from_log(log_content):
    """从日志文本解析 Epoch"""
    if not log_content:
        return "等待中...", 0
    
    epoch_match = re.findall(r"\s+(\d+)/(\d+)\s+.*?\s+(\d+)/(\d+)\s+", log_content)
    if epoch_match:
        last = epoch_match[-1]
        try:
            cur_epoch, total_epochs = int(last[0]), int(last[1])
            cur_batch, total_batches = int(last[2]), int(last[3])
            pct = ((cur_epoch - 1) / total_epochs) * 100 + (cur_batch / total_batches) * (100 / total_epochs)
            pct = min(pct, 100)
            return f"Epoch {cur_epoch}/{total_epochs} ({cur_batch}/{total_batches} batches, {pct:.1f}%)", pct
        except:
            pass
    
    epoch_only = re.findall(r"\s+(\d+)/(\d+)\s+", log_content)
    if epoch_only:
        last = epoch_only[-1]
        try:
            cur, total = int(last[0]), int(last[1])
            pct = (cur / total) * 100
            return f"Epoch {cur}/{total} ({pct:.1f}%)", pct
        except:
            pass
    return "训练进行中...", 5


def get_status():
    data_source = "未知"
    epoch_info = "等待中..."
    progress = 0
    train_metrics = {}
    log_tail = "等待训练开始..."
    running = False
    
    latest_train_dir = get_latest_train_dir()
    
    log_content = read_log_tail(LOG_FILE, 60)
    if log_content and len(log_content.strip()) > 100:
        epoch_info, progress = parse_epoch_from_log(log_content)
        log_tail = log_content
        data_source = "bilevel_run.log"
        running = True
    elif latest_train_dir:
        csv_epoch, csv_progress, csv_metrics = parse_results_csv(latest_train_dir)
        if csv_epoch:
            epoch_info = csv_epoch
            progress = csv_progress
            train_metrics = csv_metrics
            data_source = "results.csv"
            running = True
        
        args_path = os.path.join(latest_train_dir, "args.yaml")
        if os.path.exists(args_path):
            log_tail = f"训练目录: {os.path.basename(latest_train_dir)}\n"
            log_tail += f"最后更新: {time.ctime(os.path.getmtime(latest_train_dir))}\n\n"
            try:
                with open(args_path, "r", encoding="utf-8") as f:
                    log_tail += "=== 训练配置 (args.yaml) ===\n"
                    for line in f.readlines()[:30]:
                        log_tail += line
            except:
                pass
        else:
            log_tail = f"训练目录: {os.path.basename(latest_train_dir)}\n训练进行中..."
    
    console_log = read_log_tail(CONSOLE_LOG, 20)
    if console_log and "LLM (Ollama)" in console_log:
        if log_tail == "等待训练开始...":
            log_tail = console_log
    
    experiments = parse_results_tsv()
    
    best_cds = "—"
    baseline_cds = "—"
    best_params = {}
    
    for exp in experiments:
        if exp["status"] == "BEST":
            best_cds = exp["cds"]
            break
    
    if experiments and len(experiments) > 0:
        baseline_cds = experiments[-1]["cds"] if experiments[-1]["cds"] != "-" else "—"
    
    phase = "基线实验"
    phase_sub = "Phase 0"
    exp_count = len(experiments)
    
    if exp_count >= 1:
        phase = "优化循环"
        phase_sub = f"已完成 {exp_count} 个实验"
    
    if not experiments and running:
        phase = "基线实验"
        phase_sub = "Phase 0 - 训练中"
    
    engine = "LLM (Ollama)"
    model_info = "qwen2.5-coder:7b"
    
    ollama_log = read_log_tail(CONSOLE_LOG, 15) or ""
    if "LLM (Ollama): ✅" in ollama_log:
        engine = "LLM (Ollama) ✅"
    elif "LLM (Ollama): ❌" in ollama_log:
        engine = "贝叶斯优化"
        model_info = "skopt fallback"
    
    default_params = {
        "LR0": "0.002", "MOSAIC": "1.0", 
        "BATCH": "4", "IMGSZ": "960",
        "MIXUP": "0.15", "BOX": "7.5"
    }
    
    return {
        "phase": phase,
        "phase_sub": phase_sub,
        "exp_count": exp_count,
        "best_cds": best_cds,
        "baseline_cds": baseline_cds,
        "engine": engine,
        "model_info": model_info,
        "epoch_info": epoch_info,
        "progress": min(progress, 100),
        "experiments": experiments,
        "best_params": best_params if best_params else default_params,
        "log_tail": log_tail,
        "train_metrics": train_metrics,
        "data_source": data_source,
        "running": running,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/status")
def api_status():
    return jsonify(get_status())


def run_server(port=8080):
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    PORT = int(os.environ.get("SUPERVISOR_PORT", "8080"))
    print(f"🚀 Supervisor Dashboard starting on http://127.0.0.1:{PORT}")
    run_server(PORT)

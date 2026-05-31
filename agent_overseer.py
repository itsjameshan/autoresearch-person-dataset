"""
agent_overseer.py — 智能体总管 (v2 — 内置 Web UI)

职责:
  1. 启动并监控手下全部智能体的运行状态
  2. 核心战斗群 (Orchestrator) 挂了自动重启
  3. 监控团队 (Supervisor / ArchitectureSupervisor) 挂了自动重启
  4. 持续检查训练目标是否达成 (CDS >= target_cds)
  5. 内置 Web UI (默认端口 5050) — 实时查看所有智能体状态
  6. 优雅停机 — 一个信号让所有智能体有序退出

运行方式:
  python agent_overseer.py
  python agent_overseer.py --target-cds 0.85 --max-hours 48
  python agent_overseer.py --web-port 5050 --no-v2-dashboard

架构:
  智能体总管 (AgentOverseer) — 浏览器打开 http://localhost:5050
  ├── 主战斗群：Orchestrator (Researcher + Curator + Triage 一体化)
  ├── 监督团队：SupervisorAgent (监控 + 审计 + 硬件安全 + 异常检测)
  ├── 合规团队：ArchitectureSupervisorAgent (架构合规检查)
  ├── 可视化(可选)：v2 Dashboard (只读仪表盘，端口 5052)
  └── 总管自带 Web UI：实时状态 + 日志 + 控制面板
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
import traceback
import collections
from datetime import datetime
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.join(PROJECT_ROOT, "autoresearch_v2")
PYTHON = sys.executable

TARGET_CDS = 0.85
MAX_HOURS = 48
HEALTH_CHECK_INTERVAL = 30
RESTART_BACKOFF_INITIAL = 5
RESTART_BACKOFF_MAX = 300
RESTART_BACKOFF_RESET = 180
DEFAULT_WEB_PORT = 5050

STATE_DB = os.path.join(V2_ROOT, "state", "state.db")

_LOG_BUFFER_MAX = 500
_log_buffer = collections.deque(maxlen=_LOG_BUFFER_MAX)
_log_lock = threading.Lock()


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {
        "INFO": "[总管]", "WARN": "[总管⚠]", "CRIT": "[总管🚨]",
        "OK": "[总管✅]", "START": "[总管▶]", "STOP": "[总管■]",
        "HEALTH": "[总管💓]", "WEB": "[总管🌐]",
    }.get(level, "[总管]")
    line = f"{ts} {prefix} {msg}"
    with _log_lock:
        _log_buffer.append({"ts": ts, "level": level, "msg": msg, "line": line})
    print(line)
    sys.stdout.flush()


class ManagedAgent:
    def __init__(self, name: str, role: str, cmd: list[str],
                 severity: str = "critical",
                 cwd: str = PROJECT_ROOT,
                 env: dict = None):
        self.name = name
        self.role = role
        self.cmd = cmd
        self.severity = severity
        self.cwd = cwd
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.last_restart_time = 0.0
        self.last_backoff = 0.0
        self.start_time: Optional[float] = None
        self.status = "pending"
        self.last_error = ""

    def launch(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.update(self.env)

        log(f"启动 [{self.name}] ({self.role})", "START")
        try:
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.start_time = time.time()
            self.status = "running"
            self.last_error = ""
            self._start_log_reader()
            return True
        except Exception as e:
            log(f"启动 [{self.name}] 失败: {e}", "CRIT")
            self.status = "failed"
            self.last_error = str(e)
            return False

    def _start_log_reader(self):
        def reader():
            if self.process is None or self.process.stdout is None:
                return
            try:
                for line in self.process.stdout:
                    line = line.rstrip()
                    if line:
                        log(f"[{self.name}] {line}")
            except Exception:
                pass

        t = threading.Thread(target=reader, daemon=True)
        t.start()

    def is_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self, timeout: int = 15):
        if self.process is None:
            return
        log(f"停止 [{self.name}]...", "STOP")
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    log(f"[{self.name}] 未响应，强制杀死", "WARN")
                    self.process.kill()
                    self.process.wait(timeout=5)
        except Exception as e:
            log(f"停止 [{self.name}] 时出错: {e}", "WARN")
            self.last_error = str(e)
        self.status = "stopped"

    def get_backoff(self) -> float:
        now = time.time()
        if now - self.last_backoff > RESTART_BACKOFF_RESET:
            self.restart_count = 0
        backoff = min(
            RESTART_BACKOFF_INITIAL * (2 ** self.restart_count),
            RESTART_BACKOFF_MAX,
        )
        return backoff

    def record_restart(self):
        self.restart_count += 1
        self.last_restart_time = time.time()
        self.last_backoff = time.time()

    def summary(self) -> dict:
        uptime_sec = 0
        if self.start_time and self.status == "running":
            uptime_sec = int(time.time() - self.start_time)
        h, remainder = divmod(uptime_sec, 3600)
        m, s = divmod(remainder, 60)
        return {
            "name": self.name,
            "role": self.role,
            "severity": self.severity,
            "status": self.status,
            "pid": self.process.pid if self.process else None,
            "restart_count": self.restart_count,
            "uptime_sec": uptime_sec,
            "uptime": f"{h}h{m:02d}m{s:02d}s",
            "last_error": self.last_error,
        }


class HealthChecker:
    def __init__(self, state_db_path: str = STATE_DB):
        self.state_db_path = state_db_path
        self.last_known_cds = -1.0
        self.last_known_experiment_count = 0
        self.check_count = 0

    def check(self) -> dict:
        self.check_count += 1
        result = {
            "db_accessible": False,
            "best_cds": self.last_known_cds,
            "best_metrics": {},
            "experiment_count": self.last_known_experiment_count,
            "stagnation": False,
        }

        if not os.path.exists(self.state_db_path):
            return result

        import sqlite3
        try:
            conn = sqlite3.connect(self.state_db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) as cnt FROM experiments")
            row = cur.fetchone()
            if row:
                result["experiment_count"] = row["cnt"]
                self.last_known_experiment_count = row["cnt"]

            cur.execute("""
                SELECT * FROM experiments
                WHERE status = 'completed'
                ORDER BY
                    CAST(json_extract(metrics_json, '$.cds') AS REAL) DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                try:
                    m = json.loads(row["metrics_json"] or "{}")
                except Exception:
                    m = {}
                result["best_metrics"] = m
                cds = m.get("cds", 0)
                if cds > self.last_known_cds:
                    self.last_known_cds = cds
                result["best_cds"] = self.last_known_cds
                result["db_accessible"] = True

            conn.close()
        except Exception as e:
            log(f"健康检查读取状态库失败: {e}", "WARN")

        if self.check_count > 10 and self.last_known_cds >= 0:
            result["stagnation"] = self.last_known_cds == result.get("best_cds", 0)

        return result

    def full_snapshot(self) -> dict:
        result = self.check()
        result["db_path"] = self.state_db_path
        result["db_exists"] = os.path.exists(self.state_db_path)
        return result


_PAGE_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>智能体总管 — AutoResearch Overseer</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d2991d;
    --purple: #a371f7; --orange: #db6d28;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh;
  }
  header {
    background: var(--card); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; justify-content: space-between;
    align-items: center; position: sticky; top: 0; z-index: 10;
  }
  header h1 { font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
  header h1 .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .dot-ok { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot-warn { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }
  .dot-err { background: var(--red); box-shadow: 0 0 6px var(--red); animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  header .meta { font-size: 12px; color: var(--muted); }
  main {
    max-width: 1500px; margin: 0 auto; padding: 20px;
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 18px 20px;
  }
  .card.full { grid-column: 1 / -1; }
  .card h2 {
    font-size: 13px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 14px; display: flex; justify-content: space-between;
    align-items: center;
  }
  .card h2 .badge {
    font-size: 11px; padding: 2px 8px; border-radius: 10px;
    background: var(--border); color: var(--muted);
    text-transform: none; letter-spacing: 0; font-weight: 500;
  }
  .card h2 .badge.live {
    background: rgba(63,185,80,0.15); color: var(--green);
  }

  .stat-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
  }
  .stat {
    background: var(--bg); padding: 12px; border-radius: 6px;
    border: 1px solid var(--border);
  }
  .stat .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
  .stat .value { font-size: 26px; font-weight: 700; margin: 4px 0; }
  .stat .sub { font-size: 11px; color: var(--muted); }
  .val-ok { color: var(--green); }
  .val-warn { color: var(--yellow); }
  .val-err { color: var(--red); }
  .val-accent { color: var(--accent); }

  .progress-bar {
    height: 6px; background: var(--border); border-radius: 3px;
    margin-top: 8px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: var(--accent); border-radius: 3px;
    transition: width 1s ease;
  }

  .agent-card {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 12px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; margin-bottom: 10px;
  }
  .agent-card:last-child { margin-bottom: 0; }
  .agent-status-dot {
    width: 14px; height: 14px; border-radius: 50%; margin-top: 3px;
    flex-shrink: 0;
  }
  .agent-info { flex: 1; min-width: 0; }
  .agent-name { font-weight: 600; font-size: 14px; display: flex; gap: 8px; align-items: center; }
  .agent-role { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .agent-meta {
    font-size: 11px; color: var(--muted); margin-top: 4px;
    display: flex; gap: 14px; flex-wrap: wrap;
  }
  .agent-actions { display: flex; gap: 6px; flex-shrink: 0; }
  .agent-actions button {
    font-size: 11px; padding: 4px 10px; border-radius: 4px;
    border: 1px solid var(--border); background: var(--card);
    color: var(--text); cursor: pointer; transition: 0.15s;
    white-space: nowrap;
  }
  .agent-actions button:hover { border-color: var(--accent); color: var(--accent); }
  .agent-actions button.danger:hover { border-color: var(--red); color: var(--red); }

  .severity-critical { border-left: 3px solid var(--red); }
  .severity-warning { border-left: 3px solid var(--yellow); }
  .severity-info { border-left: 3px solid var(--accent); }

  .logs {
    height: 380px; overflow-y: auto; font-family: "Cascadia Code", "Fira Code", ui-monospace, Menlo, monospace;
    font-size: 12px; line-height: 1.6; background: var(--bg);
    padding: 10px 14px; border-radius: 6px; border: 1px solid var(--border);
  }
  .logs .line { padding: 1px 0; white-space: pre-wrap; word-break: break-word; }
  .logs .line.warn { color: var(--yellow); }
  .logs .line.crit { color: var(--red); }
  .logs .line.ok { color: var(--green); }
  .logs .line.start { color: var(--purple); }

  .btn-row {
    display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;
  }
  .btn-row button {
    padding: 7px 16px; border-radius: 6px; font-size: 13px;
    border: 1px solid var(--border); background: var(--card);
    color: var(--text); cursor: pointer; transition: 0.15s;
  }
  .btn-row button:hover:not(:disabled) { border-color: var(--accent); }
  .btn-row button:disabled { opacity: 0.4; cursor: default; }
  .btn-row button.primary { background: rgba(88,166,255,0.15); border-color: var(--accent); color: var(--accent); }
  .btn-row button.danger-btn { background: rgba(248,81,73,0.1); border-color: rgba(248,81,73,0.4); color: var(--red); }

  .empty-state {
    text-align: center; padding: 30px; color: var(--muted); font-size: 13px;
  }

  footer { text-align: center; color: var(--muted); font-size: 11px; padding: 16px; }
  .agent-crash-history { font-size: 11px; color: var(--red); }

  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>
    <span id="overseer-dot" class="dot dot-ok"></span>
    智能体总管 · AutoResearch
  </h1>
  <div class="meta" id="header-meta">启动中…</div>
</header>

<main>

<section class="card full">
  <h2>训练概览 <span class="badge live" id="refreshed">auto</span></h2>
  <div class="stat-row">
    <div class="stat">
      <div class="label">最佳 CDS</div>
      <div class="value" id="best-cds">—</div>
      <div class="sub" id="cds-detail"></div>
      <div class="progress-bar"><div class="progress-fill" id="cds-progress" style="width:0%"></div></div>
    </div>
    <div class="stat">
      <div class="label">目标达成</div>
      <div class="value" id="goal-met">—</div>
      <div class="sub" id="goal-detail"></div>
    </div>
    <div class="stat">
      <div class="label">实验总数</div>
      <div class="value val-accent" id="exp-count">—</div>
      <div class="sub" id="db-status"></div>
    </div>
    <div class="stat">
      <div class="label">总管运行</div>
      <div class="value val-accent" id="overseer-uptime">—</div>
      <div class="sub" id="target-info"></div>
    </div>
    <div class="stat" id="metric-p-stat" style="display:none">
      <div class="label">Precision</div>
      <div class="value" id="metric-p">—</div>
    </div>
    <div class="stat" id="metric-r-stat" style="display:none">
      <div class="label">Recall</div>
      <div class="value" id="metric-r">—</div>
    </div>
    <div class="stat" id="metric-map50-stat" style="display:none">
      <div class="label">mAP50</div>
      <div class="value" id="metric-map50">—</div>
    </div>
    <div class="stat" id="metric-cmae-stat" style="display:none">
      <div class="label">Count MAE</div>
      <div class="value" id="metric-cmae">—</div>
    </div>
  </div>
</section>

<section class="card full">
  <h2>手下智能体 <span class="badge" id="agent-count">0</span></h2>
  <div id="agents-container"></div>
  <div class="btn-row">
    <button class="primary" onclick="actionAll('restart')">全部重启</button>
    <button class="danger-btn" onclick="actionAll('stop')">全部停止</button>
  </div>
</section>

<section class="card full">
  <h2>实时日志 <span class="badge live" id="log-count">0</span></h2>
  <div class="logs" id="logs-container"><div class="empty-state">等待日志…</div></div>
</section>

</main>

<footer>智能体总管 · 自动刷新 5s · 训练不停，总管不休</footer>

<script>
const WEB_PORT = location.port;

function fmt(v, digits) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    if (digits !== undefined) return v.toFixed(digits);
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(4);
  }
  return String(v);
}

function statusClass(s) {
  if (s === "running") return "dot-ok";
  if (s === "crashed" || s === "failed") return "dot-err";
  return "dot-warn";
}

function logLevelClass(l) {
  if (l === "WARN") return "warn";
  if (l === "CRIT") return "crit";
  if (l === "OK" || l === "START") return "ok";
  return "";
}

function agentSeverityClass(s) {
  return "severity-" + (s || "info");
}

function refresh() {
  fetch("/api/status").then(r => r.json()).then(d => {
    const h = d.health || {};
    const best = h.best_metrics || {};

    const cds = h.best_cds >= 0 ? h.best_cds : 0;
    document.getElementById("best-cds").textContent = cds > 0 ? cds.toFixed(4) : "—";
    document.getElementById("cds-detail").textContent =
      best.run_id ? "run " + String(best.run_id).slice(0,8) : "";
    const target = d.target_cds || 0.85;
    const pct = Math.min(100, (cds / target) * 100);
    document.getElementById("cds-progress").style.width = pct + "%";

    const met = cds >= target && h.experiment_count >= 1;
    const gmEl = document.getElementById("goal-met");
    gmEl.textContent = met ? "达成!" : "进行中";
    gmEl.className = "value " + (met ? "val-ok" : "val-warn");
    document.getElementById("goal-detail").textContent = "目标: " + target.toFixed(2);

    document.getElementById("exp-count").textContent = h.experiment_count;
    document.getElementById("db-status").textContent =
      h.db_accessible ? "DB在线" : (h.db_exists ? "DB不可达" : "等待创建");

    document.getElementById("overseer-uptime").textContent = d.overseer_uptime || "—";
    document.getElementById("target-info").textContent = "最长 " + (d.max_hours || 48) + "h";

    if (best.precision !== undefined) {
      document.getElementById("metric-p-stat").style.display = "";
      document.getElementById("metric-p").textContent = best.precision.toFixed(4);
      document.getElementById("metric-r-stat").style.display = "";
      document.getElementById("metric-r").textContent = best.recall.toFixed(4);
      document.getElementById("metric-map50-stat").style.display = "";
      document.getElementById("metric-map50").textContent = best.mAP50 ? best.mAP50.toFixed(4) : "—";
    }
    if (best.counting_mae !== undefined) {
      document.getElementById("metric-cmae-stat").style.display = "";
      document.getElementById("metric-cmae").textContent = best.counting_mae.toFixed(2);
    }

    document.getElementById("refreshed").textContent =
      new Date().toLocaleTimeString();

    const agents = d.agents || [];
    document.getElementById("agent-count").textContent = agents.length;
    renderAgents(agents);

    let overallOk = agents.every(a => a.status === "running");
    let anyErr = agents.some(a => a.status === "crashed" || a.status === "failed");
    const dot = document.getElementById("overseer-dot");
    dot.className = "dot " + (anyErr ? "dot-err" : overallOk ? "dot-ok" : "dot-warn");

    document.getElementById("header-meta").textContent =
      "端口 " + WEB_PORT + " · 总运行 " + (d.overseer_uptime || "—");
  });

  fetch("/api/logs?limit=200").then(r => r.json()).then(d => {
    const logs = d.logs || [];
    document.getElementById("log-count").textContent = logs.length;
    const el = document.getElementById("logs-container");
    if (!logs.length) { el.innerHTML = '<div class="empty-state">暂无日志</div>'; return; }
    el.innerHTML = logs.map(l =>
      '<div class="line ' + logLevelClass(l.level) + '">' +
      escapeHtml(l.line || (l.ts + " " + l.msg)) + '</div>'
    ).join("");
    el.scrollTop = el.scrollHeight;
  });
}

function renderAgents(agents) {
  const el = document.getElementById("agents-container");
  if (!agents.length) {
    el.innerHTML = '<div class="empty-state">尚无注册的智能体</div>';
    return;
  }
  el.innerHTML = agents.map(a =>
    '<div class="agent-card ' + agentSeverityClass(a.severity) + '">' +
      '<div class="agent-status-dot ' + statusClass(a.status) + '"></div>' +
      '<div class="agent-info">' +
        '<div class="agent-name">' +
          escapeHtml(a.name) +
          (a.severity === "critical" ? ' <span style="color:var(--red);font-size:10px">核心</span>' : '') +
          (a.severity === "warning" ? ' <span style="color:var(--yellow);font-size:10px">监督</span>' : '') +
        '</div>' +
        '<div class="agent-role">' + escapeHtml(a.role) + '</div>' +
        '<div class="agent-meta">' +
          '<span>状态: <b>' + escapeHtml(a.status) + '</b></span>' +
          '<span>PID: ' + (a.pid || "—") + '</span>' +
          '<span>运行: ' + (a.uptime || "—") + '</span>' +
          (a.restart_count > 0 ? '<span class="agent-crash-history">重启次数: ' + a.restart_count + '</span>' : '') +
          (a.last_error ? '<span style="color:var(--red)">错误: ' + escapeHtml(a.last_error.substring(0,80)) + '</span>' : '') +
        '</div>' +
      '</div>' +
      '<div class="agent-actions">' +
        (a.status !== "running" ? '<button onclick="agentAction(\'' + escapeHtml(a.name) + '\',\'start\')">启动</button>' : '') +
        (a.status === "running" ? '<button class="danger" onclick="agentAction(\'' + escapeHtml(a.name) + '\',\'stop\')">停止</button>' : '') +
        '<button onclick="agentAction(\'' + escapeHtml(a.name) + '\',\'restart\')">重启</button>' +
      '</div>' +
    '</div>'
  ).join("");
}

function agentAction(name, action) {
  if (action === "stop" && !confirm("确定要停止 " + name + " 吗？")) return;
  fetch("/api/agent/" + encodeURIComponent(name) + "/" + action, { method: "POST" })
    .then(r => r.json()).then(d => { setTimeout(refresh, 2000); })
    .catch(e => alert("操作失败: " + e));
}

function actionAll(action) {
  if (action === "stop" && !confirm("确定要停止全部智能体吗？这将终止训练！")) return;
  fetch("/api/agents/" + action, { method: "POST" })
    .then(r => r.json()).then(d => { setTimeout(refresh, 2000); });
}

function escapeHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class AgentOverseer:
    def __init__(self, config: dict = None):
        config = config or {}

        self.target_cds = config.get("target_cds", TARGET_CDS)
        self.max_hours = config.get("max_hours", MAX_HOURS)
        self.health_interval = config.get("health_interval", HEALTH_CHECK_INTERVAL)
        self.enable_v2_dashboard = config.get("enable_v2_dashboard", True)
        self.enable_arch_supervisor = config.get("enable_arch_supervisor", True)
        self.enable_supervisor = config.get("enable_supervisor", True)
        self.auto_approve_hitl = config.get("auto_approve_hitl", True)
        self.data_yaml = config.get("data_yaml", os.path.join(PROJECT_ROOT, "person_dataset", "person.yaml"))
        self.imgsz = config.get("imgsz", 1280)
        self.llm_backend = config.get("llm_backend", "auto")
        self.ollama_model = config.get("ollama_model", "gemma3:4b")
        self.max_iterations = config.get("max_iterations", 50)
        self.web_port = config.get("web_port", DEFAULT_WEB_PORT)
        self.web_host = config.get("web_host", "0.0.0.0")

        self.agents: list[ManagedAgent] = []
        self.health = HealthChecker(STATE_DB)
        self.start_time: Optional[float] = None
        self.running = True

        self.stop_signal_file = os.path.join(V2_ROOT, "state", "STOP")

        self._register_agents()

    def _register_agents(self):
        os.makedirs(os.path.join(V2_ROOT, "state"), exist_ok=True)

        orchestrator_cmd = [
            PYTHON, "-u", "-m", "autoresearch_v2.orchestrator",
            "--target-cds", str(self.target_cds),
            "--max-iterations", str(self.max_iterations),
            "--llm-backend", self.llm_backend,
            "--ollama-model", self.ollama_model,
            "--data-yaml", self.data_yaml,
            "--imgsz", str(self.imgsz),
            "--cooldown", "10",
        ]
        if not self.auto_approve_hitl:
            orchestrator_cmd.append("--manual-hitl")

        self.agents.append(ManagedAgent(
            name="Orchestrator",
            role="主战斗群: Researcher + Curator + Triage 一体化闭环",
            cmd=orchestrator_cmd,
            severity="critical",
        ))

        if self.enable_supervisor:
            supervisor_cmd = [
                PYTHON, "-u", os.path.join(PROJECT_ROOT, "supervisor_agent.py"),
            ]
            self.agents.append(ManagedAgent(
                name="SupervisorAgent",
                role="监督智能体: 指标监控 + 决策审计 + 硬件安全 + 异常检测",
                cmd=supervisor_cmd,
                severity="warning",
            ))

        if self.enable_arch_supervisor:
            arch_cmd = [
                PYTHON, "-u", "-m", "autoresearch_v2.agents.architecture_supervisor",
                "--watch", "--interval", "120",
            ]
            self.agents.append(ManagedAgent(
                name="ArchitectureSupervisor",
                role="架构合规监督: 32条架构规则持续审计",
                cmd=arch_cmd,
                severity="warning",
            ))

        if self.enable_v2_dashboard:
            dashboard_cmd = [
                PYTHON, "-u", "-m", "autoresearch_v2.dashboard",
                "--port", "5052",
            ]
            self.agents.append(ManagedAgent(
                name="Dashboard-v2",
                role="v2 只读仪表盘: 实验详情 / 决策记录 / FP-FN 分析 (端口 5052)",
                cmd=dashboard_cmd,
                severity="info",
            ))

    def _find_agent(self, name: str) -> Optional[ManagedAgent]:
        for a in self.agents:
            if a.name == name:
                return a
        return None

    def start_agent(self, name: str) -> bool:
        agent = self._find_agent(name)
        if agent is None:
            return False
        if agent.is_alive():
            log(f"[{name}] 已在运行中", "WARN")
            return True
        return agent.launch()

    def stop_agent(self, name: str) -> bool:
        agent = self._find_agent(name)
        if agent is None:
            return False
        agent.stop()
        return True

    def restart_agent(self, name: str) -> bool:
        agent = self._find_agent(name)
        if agent is None:
            return False
        if agent.is_alive():
            agent.stop()
        time.sleep(2)
        return agent.launch()

    def launch_all(self) -> bool:
        log("=" * 60)
        log("  智能体总管启动 — 开始调度手下全部智能体")
        log("=" * 60)
        log(f"手下智能体: {len(self.agents)} 个")
        log(f"目标 CDS: {self.target_cds}")
        log(f"最长运行: {self.max_hours} 小时")
        log(f"健康检查间隔: {self.health_interval} 秒")
        log(f"Web UI: http://{self.web_host}:{self.web_port}")
        log("")

        all_ok = True
        for agent in self.agents:
            if not agent.launch():
                all_ok = False
                log(f"[{agent.name}] 启动失败", "CRIT")
            time.sleep(2)

        self.start_time = time.time()
        return all_ok

    def stop_all(self):
        log("正在让所有智能体下班...")
        self.running = False
        if os.path.exists(self.stop_signal_file):
            try:
                os.remove(self.stop_signal_file)
            except Exception:
                pass
        open(self.stop_signal_file, "w").close()
        time.sleep(2)

        for agent in self.agents:
            agent.stop(timeout=15)

        if os.path.exists(self.stop_signal_file):
            try:
                os.remove(self.stop_signal_file)
            except Exception:
                pass

    def _handle_agent_crash(self, agent: ManagedAgent):
        backoff = agent.get_backoff()
        agent.record_restart()
        agent.status = "crashed"

        log(
            f"[{agent.name}] 挂了 (第 {agent.restart_count} 次), "
            f"{backoff:.0f} 秒后重启...",
            "WARN" if agent.severity != "critical" else "CRIT",
        )

        if agent.severity == "critical" and agent.restart_count > 10:
            log(
                f"[{agent.name}] 已连续崩溃 {agent.restart_count} 次，"
                f"疑似系统性问题，总管将停止",
                "CRIT",
            )
            self.running = False
            return

        time.sleep(backoff)

        if not self.running:
            return

        if agent.launch():
            log(f"[{agent.name}] 重启成功", "OK")
        else:
            log(f"[{agent.name}] 重启失败", "CRIT")

    def _uptime_str(self) -> str:
        if self.start_time is None:
            return "0s"
        sec = int(time.time() - self.start_time)
        h, remainder = divmod(sec, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    def _check_goal(self) -> bool:
        h = self.health.check()
        if h["best_cds"] >= self.target_cds and h["experiment_count"] >= 1:
            log(f"目标达成！CDS={h['best_cds']:.4f} >= {self.target_cds}", "OK")
            return True
        return False

    def _check_timeout(self) -> bool:
        if self.start_time is None:
            return False
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours >= self.max_hours:
            log(f"运行时间已达上限 {self.max_hours} 小时", "WARN")
            return True
        return False

    def _persist_overseer_log(self):
        log_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "overseer_status.json")
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "uptime": self._uptime_str(),
                "agents": [a.summary() for a in self.agents],
            }
            health = self.health.check()
            status["best_cds"] = health["best_cds"]
            status["experiment_count"] = health["experiment_count"]
            status["target_cds"] = self.target_cds
            status["goal_met"] = health["best_cds"] >= self.target_cds
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _get_full_status(self) -> dict:
        return {
            "overseer_uptime": self._uptime_str(),
            "target_cds": self.target_cds,
            "max_hours": self.max_hours,
            "running": self.running,
            "agents": [a.summary() for a in self.agents],
            "health": self.health.full_snapshot(),
            "web_port": self.web_port,
        }

    def _start_web_ui(self):
        try:
            from flask import Flask, jsonify, render_template_string, request
        except ImportError:
            log("Flask 未安装，跳过 Web UI。pip install flask", "WARN")
            return

        app = Flask(__name__)
        overseer_ref = self

        @app.route("/")
        def index():
            return render_template_string(_PAGE_HTML)

        @app.route("/api/status")
        def api_status():
            return jsonify(overseer_ref._get_full_status())

        @app.route("/api/logs")
        def api_logs():
            limit = max(1, min(int(request.args.get("limit", 200)), 500))
            with _log_lock:
                logs = list(_log_buffer)[-limit:]
            return jsonify({"logs": logs, "total": len(_log_buffer)})

        @app.route("/api/health")
        def api_health():
            return jsonify({"ok": True, "uptime": overseer_ref._uptime_str()})

        @app.route("/api/agent/<name>/<action>", methods=["POST"])
        def api_agent_action(name, action):
            if action == "start":
                ok = overseer_ref.start_agent(name)
                return jsonify({"ok": ok, "action": "start", "name": name})
            elif action == "stop":
                ok = overseer_ref.stop_agent(name)
                return jsonify({"ok": ok, "action": "stop", "name": name})
            elif action == "restart":
                ok = overseer_ref.restart_agent(name)
                return jsonify({"ok": ok, "action": "restart", "name": name})
            return jsonify({"ok": False, "error": "unknown action"}), 400

        @app.route("/api/agents/<action>", methods=["POST"])
        def api_agents_action(action):
            if action == "stop":
                for a in overseer_ref.agents:
                    a.stop()
                return jsonify({"ok": True, "action": "stop_all"})
            elif action == "restart":
                for a in overseer_ref.agents:
                    if a.is_alive():
                        a.stop()
                    time.sleep(1)
                    a.launch()
                    time.sleep(1)
                return jsonify({"ok": True, "action": "restart_all"})
            return jsonify({"ok": False, "error": "unknown action"}), 400

        connect_url = f"http://127.0.0.1:{self.web_port}"
        log(f"Web UI 启动: {connect_url}", "WEB")

        def run_flask():
            try:
                app.run(
                    host=self.web_host,
                    port=self.web_port,
                    debug=False,
                    use_reloader=False,
                )
            except Exception as e:
                log(f"Web UI 异常: {e}", "CRIT")

        t = threading.Thread(target=run_flask, daemon=True)
        t.start()

        time.sleep(1.5)
        self._try_open_browser(connect_url)

    def _try_open_browser(self, url: str):
        try:
            import webbrowser
            for _ in range(5):
                try:
                    import urllib.request
                    urllib.request.urlopen(url + "/api/health", timeout=1)
                    webbrowser.open(url)
                    log(f"浏览器已打开: {url}", "OK")
                    return
                except Exception:
                    time.sleep(1)
            webbrowser.open(url)
            log(f"浏览器已打开: {url}", "WEB")
        except Exception:
            pass

    def run(self):
        self._start_web_ui()
        time.sleep(1)

        if not self.launch_all():
            log("部分智能体启动失败，但总管继续监控", "WARN")

        last_persist = 0.0
        PERSIST_INTERVAL = 120

        try:
            while self.running:
                time.sleep(self.health_interval)

                if self._check_goal():
                    self.running = False
                    break

                if self._check_timeout():
                    self.running = False
                    break

                if os.path.exists(self.stop_signal_file):
                    log("检测到 STOP 信号", "WARN")
                    self.running = False
                    break

                for agent in self.agents:
                    if not agent.is_alive() and agent.status == "running":
                        self._handle_agent_crash(agent)

                all_critical_dead = True
                for agent in self.agents:
                    if agent.severity == "critical" and agent.is_alive():
                        all_critical_dead = False
                        break
                if all_critical_dead:
                    log("所有关键智能体均已停止", "WARN")
                    self.running = False
                    break

                now = time.time()
                if now - last_persist >= PERSIST_INTERVAL:
                    self._persist_overseer_log()
                    last_persist = now

        except KeyboardInterrupt:
            log("收到中断信号", "WARN")
        except Exception as e:
            log(f"总管异常: {e}", "CRIT")
            traceback.print_exc()
        finally:
            self._print_final_report()
            self.stop_all()
            self._persist_overseer_log()

    def _print_final_report(self):
        log("")
        log("=" * 60)
        log("  智能体总管 — 最终报告")
        log("=" * 60)
        log(f"总运行时间: {self._uptime_str()}", "INFO")

        health = self.health.check()
        log(f"最佳 CDS: {health['best_cds']:.4f} (目标: {self.target_cds})", "INFO")
        if health["best_metrics"]:
            m = health["best_metrics"]
            for k in ["mAP50", "precision", "recall", "small_obj_recall", "counting_mae"]:
                if k in m:
                    log(f"最佳 {k}: {m[k]:.4f}", "INFO")
        log(f"总实验数: {health['experiment_count']}", "INFO")

        log("")
        log("  各智能体最终状态:", "INFO")
        for agent in self.agents:
            s = agent.summary()
            icon = "✅" if s["status"] == "running" else "⏸"
            log(f"    {icon} {s['name']}: {s['status']} | "
                f"重启{s['restart_count']}次 | PID={s['pid']}", "INFO")

        if health["best_cds"] >= self.target_cds:
            log("")
            log("目标达成！训练任务完成！", "OK")
        log("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="智能体总管 — 管理手下全部智能体，不停干活直到目标达成"
    )
    parser.add_argument("--target-cds", type=float, default=TARGET_CDS)
    parser.add_argument("--max-hours", type=float, default=MAX_HOURS)
    parser.add_argument("--health-interval", type=int, default=HEALTH_CHECK_INTERVAL)
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT,
                        help=f"总管 Web UI 端口 (默认: {DEFAULT_WEB_PORT})")
    parser.add_argument("--web-host", type=str, default="0.0.0.0",
                        help="Web UI 绑定地址 (默认: 0.0.0.0)")
    parser.add_argument("--llm-backend", type=str, default="auto",
                        choices=["auto", "anthropic", "ollama"])
    parser.add_argument("--ollama-model", type=str, default="gemma3:4b")
    parser.add_argument("--data-yaml", type=str,
                        default=os.path.join(PROJECT_ROOT, "person_dataset", "person.yaml"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--no-v2-dashboard", action="store_true",
                        help="不启动 v2 Dashboard (端口 5052)")
    parser.add_argument("--no-supervisor", action="store_true",
                        help="不启动 SupervisorAgent")
    parser.add_argument("--no-arch-supervisor", action="store_true",
                        help="不启动 ArchitectureSupervisor")
    parser.add_argument("--manual-hitl", action="store_true",
                        help="禁用自动审批，改为人工审批")
    args = parser.parse_args()

    config = {
        "target_cds": args.target_cds,
        "max_hours": args.max_hours,
        "health_interval": args.health_interval,
        "web_port": args.web_port,
        "web_host": args.web_host,
        "enable_v2_dashboard": not args.no_v2_dashboard,
        "enable_supervisor": not args.no_supervisor,
        "enable_arch_supervisor": not args.no_arch_supervisor,
        "auto_approve_hitl": not args.manual_hitl,
        "data_yaml": args.data_yaml,
        "imgsz": args.imgsz,
        "llm_backend": args.llm_backend,
        "ollama_model": args.ollama_model,
        "max_iterations": args.max_iterations,
    }

    overseer = AgentOverseer(config)
    overseer.run()


if __name__ == "__main__":
    main()
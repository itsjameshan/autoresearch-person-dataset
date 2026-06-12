"""
hitl.py — Human-In-The-Loop 人工审批门

两种模式:
  1. Flask 页面审批 (默认)
  2. 命令行审批 (headless)

使用方式:
  from autoresearch_v2.hitl import HITLGate
  gate = HITLGate(state)
  gate_id = gate.request(run_id, "researcher", "需要批准: 预算超30%")
  approved = gate.wait_for_decision(gate_id, timeout=3600)
"""

import json
import sys
import time
import threading
from datetime import datetime
from typing import Optional

try:
    from flask import Flask, request as flask_request, jsonify, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


HITL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>AutoResearch HITL Gate</title>
<meta charset="utf-8">
<style>
body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
.pending { border: 2px solid #f59e0b; padding: 16px; margin: 12px 0; border-radius: 8px; }
.approved { border: 2px solid #10b981; padding: 16px; margin: 12px 0; border-radius: 8px; opacity: 0.6; }
.rejected { border: 2px solid #ef4444; padding: 16px; margin: 12px 0; border-radius: 8px; opacity: 0.6; }
button { padding: 8px 24px; margin: 4px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
.approve { background: #10b981; color: white; }
.reject { background: #ef4444; color: white; }
h1 { color: #1f2937; }
.meta { color: #6b7280; font-size: 13px; }
</style>
</head>
<body>
<h1>AutoResearch HITL 审批</h1>
<div id="gates">加载中...</div>
<script>
async function loadGates() {
    const resp = await fetch('/api/gates');
    const gates = await resp.json();
    const container = document.getElementById('gates');
    if (gates.length === 0) {
        container.innerHTML = '<p>没有待审批的请求</p>';
        return;
    }
    container.innerHTML = gates.map(g => `
        <div class="${g.decision || 'pending'}">
            <div><strong>Gate #${g.gate_id}</strong> — ${g.requestor_agent}</div>
            <div class="meta">${g.requested_at}</div>
            <div><pre>${g.prompt_text || '无描述'}</pre></div>
            ${!g.decision ? `
                <button class="approve" onclick="decide(${g.gate_id}, 'approved')">批准</button>
                <button class="reject" onclick="decide(${g.gate_id}, 'rejected')">拒绝</button>
            ` : `<div>已${g.decision === 'approved' ? '批准' : '拒绝'} by ${g.decided_by}</div>`}
        </div>
    `).join('');
}
async function decide(gateId, decision) {
    await fetch('/api/gates/' + gateId, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({decision, decided_by: 'human_web'})
    });
    loadGates();
}
loadGates();
setInterval(loadGates, 5000);
</script>
</body>
</html>
"""


class HITLGate:
    def __init__(self, state, flask_port: int = 8765):
        self.state = state
        self.flask_port = flask_port
        self._app = None
        self._server_thread = None

    def request(self, run_id: str, requestor_agent: str,
                prompt_text: str = "") -> int:
        gate_id = self.state.insert_hitl_gate(run_id, requestor_agent, prompt_text)
        return gate_id

    def wait_for_decision(self, gate_id: int, timeout: float = 3600.0,
                          poll_interval: float = 5.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            gates = self.state.get_pending_hitl_gates()
            pending_ids = {g["gate_id"] for g in gates}
            if gate_id not in pending_ids:
                row = self.state.conn.execute(
                    "SELECT decision FROM hitl_gates WHERE gate_id=?", (gate_id,)
                ).fetchone()
                if row and row["decision"] == "approved":
                    return True
                return False
            time.sleep(poll_interval)
        return False

    def approve_from_cli(self, gate_id: int, decided_by: str = "human_cli"):
        self.state.resolve_hitl_gate(gate_id, "approved", decided_by)

    def reject_from_cli(self, gate_id: int, decided_by: str = "human_cli"):
        self.state.resolve_hitl_gate(gate_id, "rejected", decided_by)

    def start_flask_server(self):
        if not FLASK_AVAILABLE:
            print("hitl: Flask not available, CLI mode only", file=sys.stderr)
            return

        app = Flask(__name__)

        @app.route("/")
        def index():
            return render_template_string(HITL_TEMPLATE)

        @app.route("/api/gates")
        def api_gates():
            pending = self.state.get_pending_hitl_gates()
            all_gates = [dict(r) for r in self.state.conn.execute(
                "SELECT * FROM hitl_gates ORDER BY requested_at DESC LIMIT 50"
            ).fetchall()]
            return jsonify(all_gates)

        @app.route("/api/gates/<int:gate_id>", methods=["POST"])
        def api_decide(gate_id):
            data = flask_request.get_json(force=True)
            decision = data.get("decision", "approved")
            decided_by = data.get("decided_by", "human_web")
            self.state.resolve_hitl_gate(gate_id, decision, decided_by)
            return jsonify({"status": "ok"})

        self._app = app
        self._server_thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=self.flask_port,
                                   debug=False, use_reloader=False),
            daemon=True,
        )
        self._server_thread.start()
        print(f"hitl: Flask server started on port {self.flask_port}")

    def list_pending_cli(self):
        gates = self.state.get_pending_hitl_gates()
        if not gates:
            print("没有待审批的请求")
            return
        for g in gates:
            print(f"  Gate #{g['gate_id']} | Agent: {g['requestor_agent']} | "
                  f"Time: {g['requested_at']}")
            if g.get("prompt_text"):
                print(f"    {g['prompt_text']}")

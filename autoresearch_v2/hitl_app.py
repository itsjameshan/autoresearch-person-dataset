"""
hitl_app.py - standalone HITL approval UI/API for autoresearch v2.

Run:
    python -m autoresearch_v2.hitl_app

Default URL:
    http://127.0.0.1:5051/
"""

from __future__ import annotations

import argparse
from typing import Any

from flask import Flask, jsonify, render_template_string, request

from autoresearch_v2.db import DB_PATH as DEFAULT_DB_PATH
from autoresearch_v2.db import StateDB


_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AutoResearch v2 HITL</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
      background: #f4f5f7;
      color: #1f2329;
    }
    header {
      background: #1f2329;
      color: #fff;
      padding: 12px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    main {
      max-width: 1100px;
      margin: 20px auto;
      padding: 0 16px;
    }
    .card {
      background: #fff;
      border: 1px solid #e3e6ea;
      border-radius: 6px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
      margin-bottom: 14px;
      padding: 14px 16px;
    }
    .gate {
      border-left: 4px solid #d0d4dc;
      margin: 10px 0;
      padding: 10px 12px;
      background: #fafbfc;
    }
    .gate.pending { border-left-color: #d08a00; }
    .gate.approved { border-left-color: #0a7d3e; opacity: 0.82; }
    .gate.rejected { border-left-color: #c33; opacity: 0.82; }
    .meta {
      font-size: 12px;
      color: #666;
      margin-bottom: 8px;
    }
    pre {
      margin: 8px 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #fff;
      border: 1px solid #e9ecf0;
      border-radius: 4px;
      padding: 8px;
    }
    button {
      border: 0;
      border-radius: 4px;
      padding: 7px 14px;
      color: #fff;
      cursor: pointer;
      margin-right: 8px;
    }
    .approve { background: #0a7d3e; }
    .reject { background: #c33; }
    .muted { color: #777; font-size: 12px; }
    .badge {
      display: inline-block;
      font-size: 11px;
      border-radius: 10px;
      padding: 2px 8px;
      background: #e7eaef;
      color: #555;
      margin-left: 6px;
    }
  </style>
</head>
<body>
  <header>
    <div><strong>AutoResearch v2 HITL</strong><span class="badge" id="pendingCount">...</span></div>
    <div class="muted">Approve or reject gating decisions</div>
  </header>
  <main>
    <div class="card">
      <div class="muted">Auto refresh every 3s. Pending gates are shown first.</div>
      <div id="gates">Loading...</div>
    </div>
  </main>

  <script>
    async function loadGates() {
      const resp = await fetch('/api/gates?limit=100');
      const payload = await resp.json();
      const gates = payload.gates || [];
      const pendingCount = gates.filter(g => !g.decision).length;
      document.getElementById('pendingCount').textContent = pendingCount + ' pending';

      if (gates.length === 0) {
        document.getElementById('gates').innerHTML = '<div class="muted">No gate records yet.</div>';
        return;
      }

      const html = gates.map(g => {
        const cls = g.decision || 'pending';
        const run = g.run_id || '-';
        const by = g.decided_by ? ` by ${g.decided_by}` : '';
        const when = g.decided_at ? ` at ${g.decided_at}` : '';
        return `
          <div class="gate ${cls}">
            <div><strong>Gate #${g.gate_id}</strong> - ${g.requestor_agent || 'unknown'}</div>
            <div class="meta">run: ${run} | requested: ${g.requested_at || '-'} </div>
            <pre>${g.prompt_text || '(no prompt text)'}</pre>
            ${!g.decision ? `
              <button class="approve" onclick="decide(${g.gate_id}, 'approved')">Approve</button>
              <button class="reject" onclick="decide(${g.gate_id}, 'rejected')">Reject</button>
            ` : `<div class="muted">Resolved: ${g.decision}${by}${when}</div>`}
          </div>
        `;
      }).join('');
      document.getElementById('gates').innerHTML = html;
    }

    async function decide(gateId, decision) {
      await fetch(`/api/gates/${gateId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ decision, decided_by: 'human_web' })
      });
      loadGates();
    }

    loadGates();
    setInterval(loadGates, 3000);
  </script>
</body>
</html>
"""


def create_app(db_path: str = DEFAULT_DB_PATH) -> Flask:
    app = Flask(__name__)
    state = StateDB(db_path=db_path)

    @app.route("/")
    def index() -> str:
        return render_template_string(_PAGE_HTML)

    @app.route("/api/gates")
    def api_gates() -> Any:
        limit_raw = request.args.get("limit", "100")
        try:
            limit = max(1, min(int(limit_raw), 500))
        except ValueError:
            limit = 100
        rows = state.conn.execute(
            "SELECT * FROM hitl_gates ORDER BY requested_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return jsonify({"gates": [dict(r) for r in rows]})

    @app.route("/api/gates/<int:gate_id>", methods=["POST"])
    def api_decide(gate_id: int) -> Any:
        payload = request.get_json(silent=True) or {}
        decision = (payload.get("decision") or "").strip().lower()
        if decision not in {"approved", "rejected"}:
            return jsonify({"ok": False, "error": "decision must be approved/rejected"}), 400
        decided_by = (payload.get("decided_by") or "human_web").strip() or "human_web"
        state.resolve_hitl_gate(gate_id, decision, decided_by=decided_by)
        return jsonify({"ok": True, "gate_id": gate_id, "decision": decision})

    @app.route("/healthz")
    def healthz() -> Any:
        return jsonify({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoResearch v2 HITL UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    app = create_app(db_path=args.db_path)
    print(f"hitl_app: serving http://{args.host}:{args.port}/")
    print(f"  db: {args.db_path}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

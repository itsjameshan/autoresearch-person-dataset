"""Tiny Flask UI for HITL approval gates.

A single page lists pending gates with approve/deny buttons. Decisions
are recorded directly into state.db so request_approval() callers
unblock immediately.

Run:
    python -m person_dataset.autoresearch_v2.hitl_app \
        --db person_dataset/autoresearch_v2/state.db \
        --port 5051

This is a deliberately minimal, single-file UI — no auth, no fancy
state. Run on localhost only unless you add your own auth layer.
"""

from __future__ import annotations

import argparse
import os
import time

from flask import Flask, request, redirect, url_for, abort

from . import db as state_db
from . import hitl


_INDEX_TEMPLATE = """<!doctype html>
<html><head>
<meta charset='utf-8'>
<title>HITL Gates</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; max-width: 980px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
th {{ background: #f4f4f6; }}
.pending {{ background: #fff8e0; }}
.approved {{ color: #0a7;}}
.denied {{ color: #c33; }}
form.inline {{ display: inline; }}
button {{ padding: 4px 10px; margin-right: 4px; cursor: pointer; }}
.approve {{ background: #0a7; color: white; border: none; border-radius: 3px; }}
.deny {{ background: #c33; color: white; border: none; border-radius: 3px; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #f9f9fb; padding: 8px; border-radius: 4px; max-height: 220px; overflow: auto; }}
.muted {{ color: #888; font-size: 0.85em; }}
input[type=text] {{ width: 220px; padding: 4px; }}
</style>
</head><body>
<h1>HITL Gates <span class='muted'>(db: {db})</span></h1>
<p>{summary}</p>
<h2>Pending ({n_pending})</h2>
{pending_table}
<h2>Recent ({n_recent})</h2>
{recent_table}
<p class='muted'>Refreshes manually. Use CLI for bulk operations: <code>python -m person_dataset.autoresearch_v2.hitl ...</code></p>
</body></html>"""


def _render_pending_row(g: dict) -> str:
    sha = (g.get("target_sha") or "")[:8]
    age = int(time.time() - (g.get("requested_at") or time.time()))
    return f"""
<tr class='pending'>
  <td>#{g['gate_id']}</td>
  <td><code>{sha}</code></td>
  <td>{g.get('requestor_agent', '?')}</td>
  <td>{age}s ago</td>
  <td><pre>{(g.get('prompt_text') or '').replace('<','&lt;')}</pre></td>
  <td>
    <form class='inline' method='post' action='/decide/{g['gate_id']}/approved'>
      <input type='text' name='comment' placeholder='comment (optional)'>
      <input type='hidden' name='decided_by' value='hitl_app'>
      <button class='approve' type='submit'>Approve</button>
    </form>
    <form class='inline' method='post' action='/decide/{g['gate_id']}/denied'>
      <input type='text' name='comment' placeholder='reason (optional)'>
      <input type='hidden' name='decided_by' value='hitl_app'>
      <button class='deny' type='submit'>Deny</button>
    </form>
  </td>
</tr>"""


def _render_recent_row(g: dict) -> str:
    sha = (g.get("target_sha") or "")[:8]
    decision = g.get("decision", "?")
    cls = "approved" if decision == "approved" else "denied" if decision == "denied" else ""
    decided_at = g.get("decided_at")
    when = ""
    if decided_at:
        ago = int(time.time() - decided_at)
        when = f"{ago}s ago"
    return f"""
<tr>
  <td>#{g['gate_id']}</td>
  <td><code>{sha}</code></td>
  <td>{g.get('requestor_agent', '?')}</td>
  <td class='{cls}'>{decision}</td>
  <td>{g.get('decided_by') or '-'}</td>
  <td>{when}</td>
  <td>{(g.get('comment') or '').replace('<','&lt;')}</td>
</tr>"""


def _render_table(headers: list[str], rows_html: str, empty: str) -> str:
    if not rows_html:
        return f"<p class='muted'>{empty}</p>"
    head = "".join(f"<th>{h}</th>" for h in headers)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table>"


def create_app(db_path: str) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        pending = hitl.list_pending(db_path)
        recent = hitl.list_recent(db_path, limit=20)
        # Filter recent to only decided ones (pending already shown above)
        recent_decided = [r for r in recent if r.get("decision") in {"approved", "denied"}]
        n_pending = len(pending)
        summary = f"{n_pending} pending"
        pending_html = "".join(_render_pending_row(g) for g in pending)
        recent_html = "".join(_render_recent_row(g) for g in recent_decided)
        return _INDEX_TEMPLATE.format(
            db=os.path.abspath(db_path),
            summary=summary,
            n_pending=n_pending,
            n_recent=len(recent_decided),
            pending_table=_render_table(
                ["#", "SHA", "Requestor", "Age", "Prompt", "Decision"],
                pending_html, "No pending gates."),
            recent_table=_render_table(
                ["#", "SHA", "Requestor", "Decision", "By", "When", "Comment"],
                recent_html, "No recent decisions."),
        )

    @app.post("/decide/<int:gate_id>/<string:decision>")
    def decide(gate_id, decision):
        if decision not in {"approved", "denied"}:
            abort(400)
        comment = request.form.get("comment") or None
        decided_by = request.form.get("decided_by") or "hitl_app"
        try:
            hitl.decide(db_path, gate_id, decision, decided_by, comment)
        except Exception as e:
            return f"failed: {e}", 500
        return redirect(url_for("index"))

    return app


def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HITL approval gate Flask UI")
    p.add_argument(
        "--db",
        default="person_dataset/autoresearch_v2/state.db",
        help="Path to state.db",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5051)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    state_db.init_db(args.db)
    app = create_app(args.db)
    print(f"hitl_app: serving http://{args.host}:{args.port}/  (db: {args.db})")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())

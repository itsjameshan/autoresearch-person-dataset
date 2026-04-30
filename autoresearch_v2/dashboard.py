"""
dashboard.py — single-page read-only dashboard for autoresearch v2.

What it shows:
  - Overview tile: best CDS, iteration count, budget remaining,
    target_met status.
  - Recent experiments table (last 20).
  - Live activity events tail (auto-refresh every 5s).
  - Pending HITL gates (with link to hitl_app for actions).
  - Recent agent decisions (researcher / curator / triage / optuna).
  - Curator data issues (from data_issues table).
  - Optuna sweep summaries (best trial per study).

Why read-only: write operations are gated by HITL elsewhere
(hitl_app on port 5051). The dashboard is for situational awareness;
it intentionally cannot KEEP/DISCARD an experiment, approve a gate,
or tweak the budget. Operators who need to act go to the dedicated
HITL UI.

Run:
    python -m autoresearch_v2.dashboard --port 5052

Then open http://localhost:5052/ on the Windows GPU machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request

# Optional state DB integration — if unavailable, the API endpoints that
# need it return empty results so the static parts of the page still work.
try:
    from autoresearch_v2.db import StateDB, DB_PATH as DEFAULT_DB_PATH
    _STATE_DB_AVAILABLE = True
except Exception:
    StateDB = None  # type: ignore
    DEFAULT_DB_PATH = "autoresearch_v2/state/state.db"
    _STATE_DB_AVAILABLE = False


DEFAULT_EVENTS_PATH = "activity_events.jsonl"


# ── HTML template (inline to keep this a single file) ───────────────────

_PAGE_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Autoresearch v2 Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
    margin: 0; padding: 0; background: #f4f5f7; color: #1f2329;
  }
  header {
    background: #1f2329; color: white; padding: 12px 24px;
    display: flex; justify-content: space-between; align-items: center;
  }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .meta { font-size: 12px; opacity: 0.7; }
  main {
    max-width: 1400px; margin: 18px auto; padding: 0 18px;
    display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
  }
  .card {
    background: white; border-radius: 6px; padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    border: 1px solid #e3e6ea;
  }
  .card.full { grid-column: 1 / -1; }
  .card h2 {
    margin: 0 0 12px 0; font-size: 14px; font-weight: 600;
    color: #555; text-transform: uppercase; letter-spacing: 0.5px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .card h2 .badge {
    background: #e7eaef; color: #555; font-size: 11px;
    padding: 2px 8px; border-radius: 10px; text-transform: none;
    letter-spacing: 0; font-weight: 500;
  }

  .stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
  }
  .stat {
    background: #f8f9fb; padding: 10px 12px; border-radius: 4px;
  }
  .stat .label {
    font-size: 11px; color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .stat .value { font-size: 22px; font-weight: 600; color: #1f2329; }
  .stat .sub { font-size: 11px; color: #6b7280; margin-top: 2px; }

  .target-met-yes { color: #0a7d3e; }
  .target-met-no { color: #c33; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td {
    padding: 6px 8px; border-bottom: 1px solid #eee; text-align: left;
    vertical-align: top;
  }
  th {
    background: #f8f9fb; font-weight: 600; color: #555;
    text-transform: uppercase; font-size: 11px; letter-spacing: 0.3px;
  }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr:hover td { background: #fafbfc; }

  .status-keep { color: #0a7d3e; font-weight: 500; }
  .status-discard { color: #999; }
  .status-failed { color: #c33; }
  .status-running { color: #0066cc; }
  .status-completed { color: #0a7d3e; }

  .events {
    height: 360px; overflow-y: auto; font-family: ui-monospace, Menlo, monospace;
    font-size: 12px; line-height: 1.5; background: #fafbfc;
    padding: 8px 12px; border-radius: 4px; border: 1px solid #e3e6ea;
  }
  .events .row {
    padding: 2px 4px; border-bottom: 1px solid transparent;
    white-space: pre-wrap; word-break: break-word;
  }
  .events .row:hover { background: #eef0f3; }
  .events .ts { color: #999; margin-right: 8px; }
  .events .et {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 11px; font-weight: 600; margin-right: 8px;
  }
  .events .et-train { background: #e0f0ff; color: #0066cc; }
  .events .et-eval { background: #f5e0ff; color: #7a2bb8; }
  .events .et-decision { background: #fff3e0; color: #b86b00; }
  .events .et-keep, .events .et-target-met { background: #d8f5dd; color: #0a7d3e; }
  .events .et-discard { background: #f0f0f0; color: #888; }
  .events .et-crash, .events .et-halt, .events .et-failed { background: #fde0e0; color: #c33; }
  .events .et-hitl { background: #fff3d0; color: #a86a00; }
  .events .et-hpo { background: #e0e8ff; color: #2547b8; }
  .events .et-curator { background: #ffe0f0; color: #b8266b; }
  .events .et-triage { background: #ffe8d0; color: #c54e00; }
  .events .et-default { background: #e7eaef; color: #555; }

  .empty { color: #999; font-style: italic; padding: 20px; text-align: center; }
  .err { color: #c33; padding: 10px; background: #fde0e0; border-radius: 4px; }

  .pill {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 500;
  }
  .pill-fp { background: #fde0e0; color: #c33; }
  .pill-fn { background: #fff3d0; color: #a86a00; }
  .pill-label { background: #e0f0ff; color: #0066cc; }
  .pill-other { background: #e7eaef; color: #555; }

  .sha { font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: #666; }
  .agent { font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
  footer { text-align: center; color: #999; font-size: 11px; padding: 20px; }
</style>
</head>
<body>
<header>
  <h1>Autoresearch v2 Dashboard</h1>
  <div class="meta">db: <code>{{ db_path }}</code> · events: <code>{{ events_path }}</code></div>
</header>

<main>

<section class="card full">
  <h2>Overview <span class="badge" id="overview-updated">…</span></h2>
  <div class="stat-grid">
    <div class="stat">
      <div class="label">Best CDS</div>
      <div class="value" id="best-cds">…</div>
      <div class="sub" id="best-sha"></div>
    </div>
    <div class="stat">
      <div class="label">Target Met</div>
      <div class="value" id="target-met">…</div>
      <div class="sub" id="target-detail"></div>
    </div>
    <div class="stat">
      <div class="label">Experiments</div>
      <div class="value" id="exp-count">…</div>
      <div class="sub" id="iteration"></div>
    </div>
    <div class="stat">
      <div class="label">Budget Remaining</div>
      <div class="value" id="budget-gpu">…</div>
      <div class="sub" id="budget-dollars"></div>
    </div>
  </div>
</section>

<section class="card full">
  <h2>Live Activity Events <span class="badge" id="events-count">0</span></h2>
  <div id="events" class="events"><div class="empty">waiting…</div></div>
</section>

<section class="card">
  <h2>Recent Experiments</h2>
  <div id="experiments"><div class="empty">…</div></div>
</section>

<section class="card">
  <h2>Recent Agent Decisions</h2>
  <div id="decisions"><div class="empty">…</div></div>
</section>

<section class="card">
  <h2>Pending HITL Gates</h2>
  <div id="hitl"><div class="empty">…</div></div>
</section>

<section class="card">
  <h2>Curator Data Issues</h2>
  <div id="issues"><div class="empty">…</div></div>
</section>

<section class="card full">
  <h2>Optuna Sweeps</h2>
  <div id="sweeps"><div class="empty">…</div></div>
</section>

</main>

<footer>autoresearch v2 · read-only · auto-refresh every 5–30s</footer>

<script>
function nowStr() { return new Date().toLocaleTimeString(); }

function safeFetch(url) {
  return fetch(url).then(r => r.ok ? r.json() : Promise.reject(r.status));
}

function escapeHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}

function fmt(v, digits) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    if (digits !== undefined) return v.toFixed(digits);
    return v.toString();
  }
  return v;
}

function shortSha(s) {
  if (!s) return "";
  return s.length > 8 ? s.substring(0, 8) : s;
}

function refreshOverview() {
  safeFetch("/api/overview").then(d => {
    const best = d.best_metrics || {};
    document.getElementById("best-cds").textContent = best.cds !== undefined ? best.cds.toFixed(4) : "—";
    document.getElementById("best-sha").textContent = best.run_id ? "run " + shortSha(best.run_id) : "";
    const tm = !!best.target_met;
    const tmEl = document.getElementById("target-met");
    tmEl.textContent = tm ? "PASS" : "—";
    tmEl.className = "value " + (tm ? "target-met-yes" : "target-met-no");
    if (best.precision !== undefined) {
      document.getElementById("target-detail").textContent =
        "P=" + best.precision.toFixed(3) + " R=" + best.recall.toFixed(3);
    }
    document.getElementById("exp-count").textContent = d.experiment_count;
    document.getElementById("iteration").textContent = d.iteration ? "iter " + d.iteration : "";
    if (d.budget && d.budget.gpu_minutes_remaining !== null) {
      const rem = d.budget.gpu_minutes_remaining;
      document.getElementById("budget-gpu").textContent = rem.toFixed(0) + " min";
      const dr = d.budget.dollars_remaining;
      document.getElementById("budget-dollars").textContent = dr !== null ? "$" + dr.toFixed(2) + " left" : "";
    } else {
      document.getElementById("budget-gpu").textContent = "—";
      document.getElementById("budget-dollars").textContent = "no budget set";
    }
    document.getElementById("overview-updated").textContent = "updated " + nowStr();
  }).catch(e => {
    document.getElementById("best-cds").textContent = "err";
  });
}

function eventClass(et) {
  if (!et) return "et-default";
  if (et.startsWith("train")) return "et-train";
  if (et.startsWith("eval")) return "et-eval";
  if (et === "decision") return "et-decision";
  if (et === "keep" || et === "target_met") return "et-keep";
  if (et === "discard") return "et-discard";
  if (et === "crash" || et === "halt" || et.endsWith("_failed")) return "et-failed";
  if (et.startsWith("hitl")) return "et-hitl";
  if (et.startsWith("hpo")) return "et-hpo";
  if (et.startsWith("curator")) return "et-curator";
  if (et.startsWith("triage")) return "et-triage";
  return "et-default";
}

function renderEvent(e) {
  const ts = e.iso || (e.ts ? new Date(e.ts * 1000).toLocaleTimeString() : "?");
  const cls = eventClass(e.event);
  const summary = e.summary || JSON.stringify(
    Object.fromEntries(Object.entries(e).filter(
      ([k]) => !["ts", "iso", "session", "event"].includes(k)
    ))
  );
  return `<div class="row">
    <span class="ts">${escapeHtml(ts)}</span>
    <span class="et ${cls}">${escapeHtml(e.event)}</span>
    ${escapeHtml(summary)}
  </div>`;
}

let lastEventTs = 0;
function refreshEvents() {
  safeFetch("/api/events?limit=80").then(d => {
    const events = d.events || [];
    const el = document.getElementById("events");
    if (events.length === 0) {
      el.innerHTML = '<div class="empty">No events yet. Start the orchestrator to populate.</div>';
      document.getElementById("events-count").textContent = "0";
      return;
    }
    el.innerHTML = events.slice().reverse().map(renderEvent).join("");
    document.getElementById("events-count").textContent = events.length + " events";
    if (events.length) {
      lastEventTs = events[events.length - 1].ts || lastEventTs;
    }
  });
}

function refreshExperiments() {
  safeFetch("/api/experiments?limit=20").then(d => {
    const rows = d.experiments || [];
    const el = document.getElementById("experiments");
    if (!rows.length) {
      el.innerHTML = '<div class="empty">no experiments yet</div>';
      return;
    }
    el.innerHTML = `<table>
      <thead><tr>
        <th>run</th><th>status</th><th class="num">CDS</th>
        <th class="num">P</th><th class="num">R</th>
        <th class="num">gpu_min</th><th>desc</th>
      </tr></thead><tbody>` +
      rows.map(r => {
        const m = r.metrics_json || {};
        const statusClass = "status-" + (r.status || "").toLowerCase().split(" ")[0];
        return `<tr>
          <td><span class="sha">${escapeHtml(shortSha(r.run_id))}</span></td>
          <td class="${statusClass}">${escapeHtml(r.status || "")}</td>
          <td class="num">${m.cds !== undefined ? m.cds.toFixed(4) : "—"}</td>
          <td class="num">${m.precision !== undefined ? m.precision.toFixed(3) : "—"}</td>
          <td class="num">${m.recall !== undefined ? m.recall.toFixed(3) : "—"}</td>
          <td class="num">${r.gpu_minutes ? r.gpu_minutes.toFixed(1) : "—"}</td>
          <td>${escapeHtml((r.description || "").slice(0, 80))}</td>
        </tr>`;
      }).join("") + "</tbody></table>";
  });
}

function refreshDecisions() {
  safeFetch("/api/decisions?limit=15").then(d => {
    const rows = d.decisions || [];
    const el = document.getElementById("decisions");
    if (!rows.length) { el.innerHTML = '<div class="empty">no decisions yet</div>'; return; }
    el.innerHTML = `<table><thead><tr>
      <th>agent</th><th>action</th><th>rationale</th><th>run</th>
    </tr></thead><tbody>` +
      rows.map(r => `<tr>
        <td><span class="agent">${escapeHtml(r.made_by_agent || "")}</span></td>
        <td>${escapeHtml(r.action_type || "")}</td>
        <td>${escapeHtml((r.rationale || "").slice(0, 100))}</td>
        <td><span class="sha">${escapeHtml(shortSha(r.run_id))}</span></td>
      </tr>`).join("") + "</tbody></table>";
  });
}

function refreshHitl() {
  safeFetch("/api/hitl").then(d => {
    const rows = d.gates || [];
    const el = document.getElementById("hitl");
    if (!rows.length) { el.innerHTML = '<div class="empty">no pending gates</div>'; return; }
    el.innerHTML = `<table><thead><tr>
      <th>id</th><th>requestor</th><th>prompt</th>
    </tr></thead><tbody>` +
      rows.map(r => `<tr>
        <td>${r.gate_id}</td>
        <td><span class="agent">${escapeHtml(r.requestor_agent || "")}</span></td>
        <td>${escapeHtml((r.prompt_text || "").slice(0, 120))}</td>
      </tr>`).join("") + "</tbody></table>" +
      '<div style="margin-top:8px;font-size:11px;color:#666">' +
      'Approve/deny via the dedicated HITL UI on port 5051: ' +
      '<a href="http://127.0.0.1:5051/" target="_blank">hitl_app</a></div>';
  });
}

function refreshIssues() {
  safeFetch("/api/issues?limit=10").then(d => {
    const rows = d.issues || [];
    const el = document.getElementById("issues");
    if (!rows.length) { el.innerHTML = '<div class="empty">no issues found yet</div>'; return; }
    el.innerHTML = `<table><thead><tr>
      <th>type</th><th>severity</th><th>description</th><th>run</th>
    </tr></thead><tbody>` +
      rows.map(r => {
        const t = (r.issue_type || "").toLowerCase();
        let cls = "pill-other";
        if (t.includes("fp") || t.includes("failure")) cls = "pill-fp";
        else if (t.includes("fn") || t.includes("missed")) cls = "pill-fn";
        else if (t.includes("label")) cls = "pill-label";
        return `<tr>
          <td><span class="pill ${cls}">${escapeHtml(r.issue_type || "")}</span></td>
          <td>${escapeHtml(r.severity || "")}</td>
          <td>${escapeHtml((r.description || "").slice(0, 100))}</td>
          <td><span class="sha">${escapeHtml(shortSha(r.discovered_in_run))}</span></td>
        </tr>`;
      }).join("") + "</tbody></table>";
  });
}

function refreshSweeps() {
  safeFetch("/api/sweeps").then(d => {
    const rows = d.sweeps || [];
    const el = document.getElementById("sweeps");
    if (!rows.length) {
      el.innerHTML = '<div class="empty">no Optuna studies found yet</div>';
      return;
    }
    el.innerHTML = `<table><thead><tr>
      <th>study</th><th class="num">trials</th><th class="num">best CDS</th>
      <th class="num">best trial</th><th>best params</th>
    </tr></thead><tbody>` +
      rows.map(r => `<tr>
        <td>${escapeHtml(r.study_name)}</td>
        <td class="num">${r.n_trials}</td>
        <td class="num">${r.best_value !== null ? r.best_value.toFixed(4) : "—"}</td>
        <td class="num">${r.best_trial !== null ? r.best_trial : "—"}</td>
        <td><code style="font-size:11px">${escapeHtml(r.best_params)}</code></td>
      </tr>`).join("") + "</tbody></table>";
  });
}

function refreshFast() {
  refreshOverview();
  refreshEvents();
  refreshHitl();
}
function refreshSlow() {
  refreshExperiments();
  refreshDecisions();
  refreshIssues();
  refreshSweeps();
}

refreshFast();
refreshSlow();
setInterval(refreshFast, 5000);
setInterval(refreshSlow, 30000);
</script>
</body>
</html>
"""


# ── Helpers ─────────────────────────────────────────────────────────────

def _events_path_resolved(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _read_events_tail(events_path: Path, limit: int = 80) -> list[dict]:
    """Read up to `limit` most recent JSONL events (cheap stream from end)."""
    if not events_path.is_file():
        return []
    try:
        # Simple tail: read all lines then take last N. Acceptable for the
        # expected event-log size (<=10k lines per session); if it grows,
        # swap to seek-based reverse read.
        with events_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        events = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
    except OSError:
        return []


def _format_event_summary(e: dict) -> str:
    """Mirror autoresearch_v2.events._format_terminal but as a server-side
    fallback so older event records (without summary) still display nicely."""
    try:
        from autoresearch_v2.events import _format_terminal
        return _format_terminal(e.get("event", ""), e)
    except Exception:
        return ""


def _budget_for_period(state: "StateDB", period: str) -> dict:
    """Combine get_budget (limits + used) with get_remaining_budget (the
    pre-clamped remaining values) into one shape suitable for the JSON API.

    StateDB.get_budget returns the raw row with gpu_minutes_limit/_used and
    dollars_limit/_used. get_remaining_budget returns clamped remaining
    values OR float('inf') when no row exists. We merge them so the
    dashboard sees a single dict with both 'used' and 'remaining' fields.
    """
    if state is None:
        return {"period": period, "exists": False}
    try:
        b = state.get_budget(period)
        if not b:
            return {"period": period, "exists": False}
        gpu_lim = b.get("gpu_minutes_limit")
        gpu_used = b.get("gpu_minutes_used") or 0.0
        dol_lim = b.get("dollars_limit")
        dol_used = b.get("dollars_used") or 0.0
        return {
            "exists": True,
            "period": period,
            "gpu_minutes_used": gpu_used,
            "gpu_minutes_limit": gpu_lim,
            "gpu_minutes_remaining": (
                max(0.0, (gpu_lim or 0.0) - gpu_used)
                if gpu_lim is not None else None
            ),
            "dollars_used": dol_used,
            "dollars_limit": dol_lim,
            "dollars_remaining": (
                max(0.0, (dol_lim or 0.0) - dol_used)
                if dol_lim is not None else None
            ),
        }
    except Exception:
        return {"period": period, "exists": False}


def _list_optuna_studies(study_db_path: Path) -> list[dict]:
    """Best-effort summary of every Optuna study in the storage SQLite.

    Returns rows with study_name, n_trials, best_value, best_trial, best_params
    string. If optuna isn't importable or the file doesn't exist, returns [].
    """
    if not study_db_path.is_file():
        return []
    try:
        import optuna
    except ImportError:
        return []

    storage = f"sqlite:///{study_db_path}"
    try:
        names = optuna.get_all_study_names(storage=storage)
    except Exception:
        return []

    rows = []
    for name in names:
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except Exception:
            continue
        n = len(study.trials)
        try:
            best = study.best_trial
            best_value = float(best.value) if best.value is not None else None
            best_trial = best.number
            best_params = json.dumps(best.params, ensure_ascii=False)
        except (ValueError, KeyError):
            best_value = None
            best_trial = None
            best_params = ""
        rows.append({
            "study_name": name,
            "n_trials": n,
            "best_value": best_value,
            "best_trial": best_trial,
            "best_params": best_params,
        })
    return rows


# ── Flask app factory ──────────────────────────────────────────────────

def create_app(
    db_path: str = DEFAULT_DB_PATH,
    events_path: str = DEFAULT_EVENTS_PATH,
    optuna_db_path: str | None = None,
    budget_period: str = "default",
) -> Flask:
    app = Flask(__name__)
    events_p = _events_path_resolved(events_path)
    optuna_p = (
        Path(optuna_db_path) if optuna_db_path
        else Path(__file__).resolve().parent / "state" / "optuna.db"
    )

    def _state() -> "StateDB | None":
        if not _STATE_DB_AVAILABLE:
            return None
        try:
            return StateDB(db_path=db_path)
        except Exception as e:
            print(f"dashboard: state DB unavailable: {e}",
                  file=sys.stderr, flush=True)
            return None

    @app.get("/")
    def index():
        return render_template_string(
            _PAGE_HTML,
            db_path=str(db_path),
            events_path=str(events_p),
        )

    @app.get("/api/overview")
    def api_overview():
        state = _state()
        best = {}
        exp_count = 0
        iteration = 0
        if state is not None:
            try:
                best = state.get_best_metrics() or {}
                exp_count = state.get_experiment_count()
                iteration = exp_count  # close enough for display
            except Exception as e:
                return jsonify({"error": str(e)}), 200
            finally:
                try: state.close()
                except Exception: pass

        return jsonify({
            "best_metrics": best,
            "experiment_count": exp_count,
            "iteration": iteration,
            "budget": _budget_for_period(_state(), budget_period),
        })

    @app.get("/api/experiments")
    def api_experiments():
        limit = max(1, min(int(request.args.get("limit", 20)), 200))
        state = _state()
        if state is None:
            return jsonify({"experiments": []})
        try:
            rows = state.get_recent_experiments(n=limit) or []
            return jsonify({"experiments": rows})
        except Exception as e:
            return jsonify({"experiments": [], "error": str(e)})
        finally:
            try: state.close()
            except Exception: pass

    @app.get("/api/decisions")
    def api_decisions():
        limit = max(1, min(int(request.args.get("limit", 20)), 200))
        state = _state()
        if state is None:
            return jsonify({"decisions": []})
        try:
            rows = state.get_decisions(limit=limit) or []
            return jsonify({"decisions": rows})
        except Exception as e:
            return jsonify({"decisions": [], "error": str(e)})
        finally:
            try: state.close()
            except Exception: pass

    @app.get("/api/issues")
    def api_issues():
        limit = max(1, min(int(request.args.get("limit", 20)), 200))
        state = _state()
        if state is None:
            return jsonify({"issues": []})
        try:
            rows = state.get_unresolved_issues(limit=limit) or []
            return jsonify({"issues": rows})
        except Exception as e:
            return jsonify({"issues": [], "error": str(e)})
        finally:
            try: state.close()
            except Exception: pass

    @app.get("/api/hitl")
    def api_hitl():
        state = _state()
        if state is None:
            return jsonify({"gates": []})
        try:
            rows = state.get_pending_hitl_gates() or []
            return jsonify({"gates": rows})
        except Exception as e:
            return jsonify({"gates": [], "error": str(e)})
        finally:
            try: state.close()
            except Exception: pass

    @app.get("/api/events")
    def api_events():
        limit = max(1, min(int(request.args.get("limit", 80)), 500))
        events = _read_events_tail(events_p, limit=limit)
        for e in events:
            if "summary" not in e:
                e["summary"] = _format_event_summary(e)
        return jsonify({"events": events, "tail_path": str(events_p)})

    @app.get("/api/sweeps")
    def api_sweeps():
        return jsonify({"sweeps": _list_optuna_studies(optuna_p)})

    @app.get("/api/health")
    def api_health():
        return jsonify({
            "ok": True,
            "db_path": str(db_path),
            "db_available": _STATE_DB_AVAILABLE,
            "events_path": str(events_p),
            "events_exists": events_p.is_file(),
            "optuna_db": str(optuna_p),
            "optuna_exists": optuna_p.is_file(),
        })

    return app


def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="state.db path")
    p.add_argument("--events", default=DEFAULT_EVENTS_PATH,
                   help="activity_events.jsonl path")
    p.add_argument("--optuna-db", default=None,
                   help="optuna.db path (default: autoresearch_v2/state/optuna.db)")
    p.add_argument("--budget-period", default="default")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5052)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    app = create_app(
        db_path=args.db,
        events_path=args.events,
        optuna_db_path=args.optuna_db,
        budget_period=args.budget_period,
    )
    print(f"dashboard: serving http://{args.host}:{args.port}/")
    print(f"  db:       {args.db}")
    print(f"  events:   {args.events}")
    print(f"  optuna:   {args.optuna_db or 'autoresearch_v2/state/optuna.db'}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())

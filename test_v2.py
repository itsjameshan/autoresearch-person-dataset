import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autoresearch_v2.db import StateDB

test_db_path = "/tmp/test_autoresearch_v2.db"
if os.path.exists(test_db_path):
    os.remove(test_db_path)

db = StateDB(test_db_path)
print("DB initialized OK")

db.insert_experiment("test_run_001", config_json={"LR0": 0.01}, status="completed")
db.update_experiment_metrics("test_run_001", {"cds": 0.75, "mAP50": 0.85})
exp = db.get_experiment("test_run_001")
print(f"Experiment: {exp['run_id']} cds={exp['metrics_json']['cds']}")

db.log_decision("test_run_001", "researcher", "hpo_sweep", "test rationale")
decisions = db.get_decisions(run_id="test_run_001")
print(f"Decisions: {len(decisions)}")

db.insert_data_issue("test.jpg", "FP", "high", "test issue")
issues = db.get_unresolved_issues()
print(f"Issues: {len(issues)}")

db.upsert_budget("2026-W18", gpu_minutes_limit=500, dollars_limit=50)
db.add_budget_usage("2026-W18", gpu_minutes=30, dollars=2)
remaining = db.get_remaining_budget("2026-W18")
print(f"Budget remaining: GPU={remaining['gpu_minutes_remaining']:.0f}min USD=${remaining['dollars_remaining']:.2f}")

db.insert_hitl_gate("test_run_001", "researcher", "test HITL")
pending = db.get_pending_hitl_gates()
print(f"Pending HITL: {len(pending)}")

best = db.get_best_metrics()
print(f"Best metrics: cds={best['cds']}")

db.close()
os.remove(test_db_path)
print("All DB operations OK!")

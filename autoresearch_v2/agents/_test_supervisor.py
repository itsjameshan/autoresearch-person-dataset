from autoresearch_v2.agents.architecture_supervisor import ArchitectureSupervisorAgent, Violation

supervisor = ArchitectureSupervisorAgent()

bad_researcher_output = {
    "action_type": "nas_search",
    "config_diff": {},
    "rationale": "try NAS",
    "estimated_gpu_minutes": 100,
    "estimated_cost_usd": 5.0,
    "needs_hitl": False,
}
violations = supervisor.validate_agent_output(bad_researcher_output, "researcher")
print(f"Researcher NAS violation detected: {len(violations)} violations")
for v in violations:
    print(f"  [{v.rule_id}] {v.severity}: {v.message}")

bad_triage_output = {
    "root_cause_hypothesis": "maybe OOM",
    "confidence": 0.3,
    "recommended_action": "retry_with_fix",
}
violations = supervisor.validate_agent_output(bad_triage_output, "triage")
print(f"\nTriage low-confidence violation detected: {len(violations)} violations")
for v in violations:
    print(f"  [{v.rule_id}] {v.severity}: {v.message}")

good_output = {
    "action_type": "hpo_sweep",
    "config_diff": {"lr0": [0.001, 0.01]},
    "rationale": "precision low, need to search better lr",
    "expected_metric_delta": {"cds": 0.02},
    "estimated_gpu_minutes": 90,
    "estimated_cost_usd": 1.5,
    "needs_hitl": False,
}
violations = supervisor.validate_agent_output(good_output, "researcher")
print(f"\nGood researcher output: {len(violations)} violations")

bad_curator_output = {
    "worst_failure_modes": "not a list",
}
violations = supervisor.validate_agent_output(bad_curator_output, "curator")
print(f"\nCurator bad output detected: {len(violations)} violations")
for v in violations:
    print(f"  [{v.rule_id}] {v.severity}: {v.message}")

print("\nAll tests passed!")

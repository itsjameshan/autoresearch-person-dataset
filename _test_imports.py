import sys, os

print('=== 完整导入测试 ===')
print()

from autoresearch_v2.db import StateDB
print('OK: db.StateDB')

from autoresearch_v2.budget import BudgetGuard, BudgetExceeded
print('OK: budget.BudgetGuard, BudgetExceeded')

from autoresearch_v2.hitl import HITLGate
print('OK: hitl.HITLGate')

from autoresearch_v2.agents.researcher import ResearcherAgent
print('OK: agents.researcher.ResearcherAgent')

from autoresearch_v2.agents.curator import CuratorAgent
print('OK: agents.curator.CuratorAgent')

from autoresearch_v2.agents.triage import TriageAgent
print('OK: agents.triage.TriageAgent')

from autoresearch_v2.tools.train_dispatcher import TrainDispatcher, TrainingFailed
print('OK: tools.train_dispatcher.TrainDispatcher, TrainingFailed')

from autoresearch_v2.tools.eval_dispatcher import EvalDispatcher
print('OK: tools.eval_dispatcher.EvalDispatcher')

from autoresearch_v2.orchestrator import Orchestrator
print('OK: orchestrator.Orchestrator')

print()
print('=== 集成测试: Orchestrator 初始化 ===')
config = {
    'llm_backend': 'ollama',
    'ollama_model': 'gemma3:4b',
    'gpu_minutes_limit': 100,
    'dollar_limit': 10,
}
orch = Orchestrator(config)
print(f'  target_cds={orch.target_cds}')
print(f'  max_iterations={orch.max_iterations}')
print(f'  researcher backend={orch.researcher.llm_backend}')
print(f'  budget gpu_limit={orch.budget.default_gpu_limit}')
print(f'  budget usd_limit={orch.budget.default_dollar_limit}')
orch.state.close()

print()
print('=== 所有导入和集成测试通过! ===')

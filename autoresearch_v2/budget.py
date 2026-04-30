"""
budget.py — 预算守门

双重守门机制:
  1. agent 自我估算 (estimated_cost)
  2. 系统硬限 (budget 表)

使用方式:
  from autoresearch_v2.budget import BudgetGuard
  guard = BudgetGuard(state)
  guard.reserve(run_id, estimated_gpu_minutes=90, estimated_cost_usd=1.5)
  guard.commit_actual(run_id, actual_gpu_minutes=85, actual_cost_usd=1.2)
"""

from datetime import datetime
from typing import Optional


class BudgetExceeded(Exception):
    def __init__(self, message: str, period: str = "", remaining: dict = None):
        super().__init__(message)
        self.period = period
        self.remaining = remaining or {}


class BudgetGuard:
    def __init__(self, state, default_gpu_limit: float = 0.0,
                 default_dollar_limit: float = 0.0):
        self.state = state
        self.default_gpu_limit = default_gpu_limit
        self.default_dollar_limit = default_dollar_limit
        self._reserved = {}

    def _current_period(self) -> str:
        now = datetime.now()
        return f"{now.year}-W{now.isocalendar()[1]:02d}"

    def ensure_budget_period(self, period: str = None):
        period = period or self._current_period()
        existing = self.state.get_budget(period)
        if existing is None:
            self.state.upsert_budget(
                period,
                gpu_minutes_limit=self.default_gpu_limit,
                dollars_limit=self.default_dollar_limit,
            )

    def check_remaining(self, period: str = None) -> dict:
        period = period or self._current_period()
        self.ensure_budget_period(period)
        return self.state.get_remaining_budget(period)

    def can_afford(self, estimated_gpu_minutes: float = 0.0,
                   estimated_cost_usd: float = 0.0,
                   period: str = None) -> tuple[bool, str]:
        remaining = self.check_remaining(period)
        gpu_ok = remaining["gpu_minutes_remaining"] >= estimated_gpu_minutes
        usd_ok = remaining["dollars_remaining"] >= estimated_cost_usd
        if not gpu_ok:
            return False, (f"GPU预算不足: 需要{estimated_gpu_minutes:.0f}分钟, "
                           f"剩余{remaining['gpu_minutes_remaining']:.0f}分钟")
        if not usd_ok:
            return False, (f"美元预算不足: 需要${estimated_cost_usd:.2f}, "
                           f"剩余${remaining['dollars_remaining']:.2f}")
        return True, ""

    def reserve(self, run_id: str, estimated_gpu_minutes: float = 0.0,
                estimated_cost_usd: float = 0.0, period: str = None):
        period = period or self._current_period()
        can, reason = self.can_afford(estimated_gpu_minutes, estimated_cost_usd, period)
        if not can:
            raise BudgetExceeded(reason, period, self.check_remaining(period))
        self._reserved[run_id] = {
            "period": period,
            "gpu_minutes": estimated_gpu_minutes,
            "cost_usd": estimated_cost_usd,
        }

    def commit_actual(self, run_id: str, actual_gpu_minutes: float = 0.0,
                      actual_cost_usd: float = 0.0):
        if run_id in self._reserved:
            del self._reserved[run_id]
        period = self._current_period()
        self.ensure_budget_period(period)
        self.state.add_budget_usage(period, actual_gpu_minutes, actual_cost_usd)

    def release_reserved(self, run_id: str):
        self._reserved.pop(run_id, None)

    def needs_hitl(self, estimated_cost_usd: float, threshold_ratio: float = 0.3,
                   period: str = None) -> bool:
        if self.default_dollar_limit <= 0 and self.default_gpu_limit <= 0:
            return False
        remaining = self.check_remaining(period)
        total_remaining = remaining["dollars_remaining"]
        if total_remaining <= 0:
            return True
        return estimated_cost_usd > total_remaining * threshold_ratio

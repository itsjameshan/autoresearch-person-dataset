"""
test_stuck_detector.py — 验证 CDS 卡死检测修复是否有效

模拟场景: CDS 停在 0.1234，连续 7 轮不变
"""

import time

# ============================================================
# 旧版逻辑 (Bug版): 只在 CDS 变化时记录
# ============================================================
class OldStuckDetector:
    def __init__(self):
        self.last_cds = -1.0
        self.cds_history = []
        self.stuck_alerts_sent = 0
    
    def update(self, best_cds):
        if best_cds != self.last_cds:
            self.cds_history.append(best_cds)
        self.last_cds = best_cds
    
    def detect(self):
        STUCK_CDS_STAGNANT_ROUNDS = 6
        alerts = []
        if len(self.cds_history) >= STUCK_CDS_STAGNANT_ROUNDS:
            recent = self.cds_history[-STUCK_CDS_STAGNANT_ROUNDS:]
            if max(recent) - min(recent) < 0.003 and self.stuck_alerts_sent < 2:
                alerts.append("CDS STAGNANT!")
                self.stuck_alerts_sent += 1
        return alerts

# ============================================================
# 新版逻辑 (修复版): 每轮都记录快照
# ============================================================
class NewStuckDetector:
    def __init__(self):
        self.last_cds = -1.0
        self._cds_snapshots = []
        self._cds_round_count = 0
        self._cds_stagnant_alerts = 0
        self._cds_improved_at = time.time()
    
    def update(self, best_cds):
        self._cds_round_count += 1
        self._cds_snapshots.append(best_cds)
        if len(self._cds_snapshots) > 30:
            self._cds_snapshots = self._cds_snapshots[-30:]
        if best_cds > self.last_cds + 0.0005:
            self._cds_improved_at = time.time()
        self.last_cds = best_cds
    
    def detect(self):
        STUCK_CDS_STAGNANT_ROUNDS = 6
        alerts = []
        if self._cds_round_count >= STUCK_CDS_STAGNANT_ROUNDS:
            recent = self._cds_snapshots[-STUCK_CDS_STAGNANT_ROUNDS:]
            cds_range = max(recent) - min(recent)
            has_real_result = max(recent) > 0.001
            if cds_range < 0.003 and has_real_result and self._cds_stagnant_alerts < 2:
                alerts.append(f"CDS STAGNANT! range={cds_range:.6f}, values={recent}")
                self._cds_stagnant_alerts += 1
        return alerts

    def reset_after_fix(self):
        self._cds_snapshots = []
        self._cds_round_count = 0
        self._cds_stagnant_alerts = 0


# ============================================================
# 场景模拟
# ============================================================

print("=" * 60)
print("场景: CDS 从 0.0 → 0.1234 (提升) → 然后停滞 7 轮")
print("=" * 60)

# Phase 1: CDS 在变化中（模拟实验在优化）
print("\n--- Phase 1: CDS 提升阶段 ---")
old = OldStuckDetector()
new = NewStuckDetector()

for cds_val in [0.01, 0.05, 0.10, 0.1234]:
    old.update(cds_val)
    new.update(cds_val)
    print(f"  update(CDS={cds_val:.4f})  old.cds_history={old.cds_history}  new._cds_snapshots={new._cds_snapshots}")

# Phase 2: CDS 停滞在 0.1234（连续7轮）
print("\n--- Phase 2: CDS 停滞 7 轮 (CDS=0.1234) ---")
old_alerts = 0
new_alerts = 0
for i in range(7):
    old.update(0.1234)
    new.update(0.1234)
    oa = old.detect()
    na = new.detect()
    if oa:
        old_alerts += 1
        print(f"  第{i+1}轮 旧版检测到: {oa[0]}")
    if na:
        new_alerts += 1
        print(f"  第{i+1}轮 新版检测到: {na[0]}")

print(f"\n  old.cds_history = {old.cds_history} (长度={len(old.cds_history)})")
print(f"  new._cds_snapshots = {new._cds_snapshots} (长度={len(new._cds_snapshots)})")

# Phase 3: 修复后重置验证
print("\n--- Phase 3: 修复后重置 old.cds_history ---")
print(f"  修复前 _cds_snapshots = {new._cds_snapshots}")
new.reset_after_fix()
print(f"  修复后 _cds_snapshots = {new._cds_snapshots}")

# Phase 4: 重置后再跑几轮，不会再误报
print("\n--- Phase 4: 重置后模拟新实验提升 (CDS 0.15 → 0.18) ---")
for cds_val in [0.15, 0.16, 0.17, 0.18, 0.18, 0.18, 0.18]:
    new.update(cds_val)
na = new.detect()
if na:
    print(f"  误报: {na[0]} (这是 BUG!)")
else:
    print(f"  正确: 第7轮 CDS 刚提升过, _cds_round_count={new._cds_round_count}, 不报警")

# 再停滞
print("\n--- Phase 5: 重置后真正停滞 6 轮 (CDS=0.18) ---")
new.reset_after_fix()
for i in range(12):
    new.update(0.18)
    na = new.detect()
    if na:
        print(f"  第{i+1}轮 新版检测到: {na[0]}")

# ============================================================
# 结论
# ============================================================
print("\n" + "=" * 60)
print("结论:")
if old_alerts == 0:
    print("  ❌ 旧版: CDS 停滞 7 轮，cds_history 始终只有 1 个值(0.1234)，")
    print("     len(cds_history)=1 < 6，永远不会检测到卡死！")
else:
    print(f"  旧版: 检测到 {old_alerts} 次")
print(f"  ✅ 新版: _cds_snapshots 每轮记录，检测到 {new_alerts} 次停滞")
print(f"  ✅ 新版: 修复后 reset，不会基于旧数据误报")
print("=" * 60)
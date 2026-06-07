# AutoResearch v2 训练日报
**日期**: 2026-06-07

## 摘要
LLM不可用，自动生成基础报告。今日完成 0 次实验，失败 97 次，最佳CDS=0.0000。

## 训练成果
- 实验总数: 100 (完成 0, 失败 97)
- 今日最佳 CDS: 0.0000
- 全局最佳 CDS: 0.4911
- CDS 提升: -0.4911
- 趋势: **unknown**
- Precision: 0.6545
- Recall: 0.6623
- mAP50: 0.6210
- mAP50-95: 0.1880
- Small Obj Recall: 0.0003
- Counting MAE: 3.25

## 错误分析
- 总错误: 97
- 未解决: 97
- 需要人工介入: 是
  - OOM: 0
  - NaN Loss: 0
  - 超时: 0
  - 配置错误: 0
  - 其他: 97
  - `optuna_hpo_20260607_183018_099c83_t007`: failed — {} (pending)
  - `optuna_hpo_20260607_183134_964eae_t002`: failed — {} (pending)
  - `optuna_hpo_20260607_182907_670125_t011`: failed — {} (pending)
  - `optuna_hpo_20260607_183018_099c83_t006`: failed — {} (pending)
  - `optuna_hpo_20260607_183134_964eae_t001`: failed — {} (pending)

## 数据问题
- 今日新增: 0
- 总计未解决: 0

## 下一步方向
- 建议动作: **escalate**
- 策略: LLM不可用，建议检查训练日志并手动评估下一步方向。
  - 风险: LLM生成报告失败，此为自动fallback报告，建议人工检查。

## Researcher 今日决策
  - `hpo_sweep`: trial=7 study=hpo_20260607_183018_099c83 params={"LR0": 0.0035482331950002504, "BOX": 14.36125290758 -> unknown
  - `hpo_sweep`: trial=11 study=hpo_20260607_182907_670125 params={"LR0": 0.006177430943619523, "BOX": 10.19711290531 -> unknown
  - `hpo_sweep`: trial=2 study=hpo_20260607_183134_964eae params={"LR0": 0.0011900590783184247, "BOX": 17.99264218662 -> unknown
  - `hpo_sweep`: trial=1 study=hpo_20260607_183134_964eae params={"LR0": 0.006009974718380315, "BOX": 7.3402796066365 -> unknown
  - `hpo_sweep`: trial=6 study=hpo_20260607_183018_099c83 params={"LR0": 0.0028265938763074048, "BOX": 13.48419636022 -> unknown
  - `hpo_sweep`: trial=10 study=hpo_20260607_182907_670125 params={"LR0": 0.0011317105459661932, "BOX": 19.7536900389 -> unknown
  - `hpo_sweep`: trial=5 study=hpo_20260607_183018_099c83 params={"LR0": 0.0018625946046852751, "BOX": 18.99292999917 -> unknown
  - `hpo_sweep`: trial=0 study=hpo_20260607_183134_964eae params={"LR0": 0.0030710573677773722, "BOX": 19.26071459614 -> unknown
  - `hpo_sweep`: trial=9 study=hpo_20260607_182907_670125 params={"LR0": 0.002163768755814362, "BOX": 10.672564313397 -> unknown
  - `hpo_sweep`: 当前仍在冷启动阶段（暂无 completed 实验），先用 Optuna 在关键参数空间做初始探索以建立基线。 -> unknown

---
*报告由 DailyReportAgent 自动生成于 2026-06-07T18:41:19.091040*
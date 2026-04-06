# autoresearch — Stadium Dense Crowd Detection

## Setup

1. Read this file (`program.md`) fully before starting.
2. Read `suggestions.md` if it exists (advisor recommendations).
3. Read `results.tsv` if it exists (experiment history).
4. Read `status.md` if it exists (current progress).
5. Verify environment: `python train.py --help` or quick import check.

Once context is loaded, run baseline first, then begin the experiment loop.

## The Goal

**Maximize CDS (Crowd Detection Score)** — a composite metric:

```
CDS = 0.30×mAP50 + 0.30×mAP50-95 + 0.10×F1 + 0.15×counting_acc + 0.15×small_obj_recall
```

Higher is better. Current baseline estimate: ~0.536.

**Quality gates** (hard targets from project requirements):
- Precision >= 0.90
- Recall >= 0.85

When BOTH gates pass in the same experiment, mark it `[TARGET_MET]`.

## What You Can Modify

**Only `train.py`** — the experiment config section at the top:
- Model: `yolov8s.pt`, `yolo12n.pt`, `yolo12s.pt`, `yolo12l.pt`
- Hyperparameters: LR, batch size, epochs, patience
- Data augmentation: mosaic, mixup, copy_paste, erasing, hsv, flips, etc.
- Loss weights: box, cls
- Close mosaic timing
- Any training parameter exposed by Ultralytics

## What You CANNOT Modify

- `evaluate.py` — read-only, defines CDS computation and quality gates
- Deployment thresholds: conf=0.25, iou_nms=0.35 (set in evaluate.py)
- CDS weights (set in evaluate.py)
- Quality gate thresholds (set in evaluate.py)

## The First Run

Your very first run is always the **baseline**. Do NOT modify `train.py`. Just run it as-is to establish the initial CDS score. Record it in results.tsv.

## Output Format

After training, the script prints:

```
---
cds:              0.5360
mAP50:            0.7100
mAP50_95:         0.2500
precision:        0.7100
recall:           0.7500
...
precision_gate:   FAIL(0.71<0.90)
recall_gate:      FAIL(0.75<0.85)
```

Extract the key metric: `grep "^cds:" run.log`

## Logging Results

Record every experiment in `results.tsv` (tab-separated):

```
commit	cds	mAP50	mAP50_95	small_obj_recall	precision	recall	counting_mae	inference_ms	memory_gb	epochs	status	description
```

- commit: 7-char git short hash
- status: `keep`, `discard`, or `crash`
- description: short text of what this experiment tried
- Add `[TARGET_MET]` to description when both quality gates pass

## Experiment Loop

### Three-Layer Safety Control

**Layer 1 — Hardware Protection (check before every run):**

```bash
# Check memory pressure
memory_pressure  # look for "normal", "warn", or "critical"
```

| State | Action |
|-------|--------|
| normal | Train normally |
| warn | Reduce BATCH to 4, continue |
| critical | Pause 5 min, recheck; if still critical → STOP |
| thermal throttle | Pause 5 min cooldown |

**Between every experiment: wait 60 seconds** (M1 Air has no fan).

**Layer 2 — Smart Stop Conditions:**

| Condition | Action |
|-----------|--------|
| 3 consecutive keeps with `[TARGET_MET]` | STOP — target achieved, push, merge to main |
| 5 consecutive discards | STOP — plateau, write analysis to suggestions.md |
| 20 total experiments this session | STOP — session cap, print progress report |
| Single run exceeds 60 minutes | Kill it, treat as crash |

**Layer 3 — Autonomous Execution:**

Run autonomously. Do NOT ask "should I continue?" between experiments. The user may be away from the computer — possibly sleeping. Each experiment: modify train.py → commit → train → evaluate → keep/discard → next.

Keep going until a stop condition is met. When you stop for any reason, always:
1. Push latest results to remote
2. Update results.tsv and status.md
3. Print a final summary of progress

### The Loop

```
AUTONOMOUS LOOP (max 20 experiments per session):

  0. Hardware health check (Layer 1)
     - critical → stop
     - warn → reduce batch size

  1. Think of a hypothesis. Read suggestions.md for advisor ideas.
  2. Modify the EXPERIMENT CONFIG section of train.py
  3. git add train.py && git commit -m "experiment: <description>"
  4. Run: python train.py > run.log 2>&1
     - If it takes > 60 min, kill it
  5. Extract: grep "^cds:\|^precision:\|^recall:" run.log
     - If grep is empty → crash. Run: tail -n 50 run.log
  6. Record results in results.tsv
  7. Update status.md (progress dashboard)

  8. Decision:
     ├─ CDS improved > 0.005 → KEEP
     │   git add results.tsv status.md
     │   git commit --amend --no-edit
     │   git push origin autoresearch/crowd-v1
     │
     ├─ CDS same or worse → DISCARD
     │   git reset --hard HEAD~1
     │   (discard counter += 1)
     │
     └─ Crash
         ├─ Simple bug (typo, import) → fix and re-run
         ├─ Fundamental issue → git reset --hard HEAD~1
         └─ Log crash in results.tsv via separate commit

  9. Check stop conditions (Layer 2)
  10. Wait 60 seconds cooldown
  11. Every 5 keeps → also run pipeline evaluation
  12. Continue to step 0
```

### Keep/Discard Threshold

CDS must improve by **more than 0.005** to be a keep. This filters out training noise.

**Simplicity criterion:** All else being equal, simpler is better. If you can remove complexity and get equal or better CDS, that's a keep. A tiny CDS improvement from ugly complex code is not worth it.

### Crashes

If a run crashes:
- Typo or missing import → fix and re-run (still counts as same experiment)
- OOM → reduce batch size or model size, try again
- Fundamental issue → skip, log as crash, move on

## Git Workflow

- Branch: `autoresearch/crowd-v1`
- Every modification to train.py → commit BEFORE running
- Keep → commit stays, push to remote
- Discard → `git reset --hard HEAD~1`
- NEVER force push
- NEVER modify evaluate.py
- NEVER touch main branch
- Run command: `python train.py > run.log 2>&1` (no tee, keep context clean)

## Exploration Priorities

Ordered by expected impact:

1. **Model upgrade**: yolo12s (bigger than n, faster than l — sweet spot for 8GB)
2. **Data augmentation**: copy_paste 0.1→0.3, mosaic tuning, close_mosaic timing
3. **Loss weights**: box loss weight tuning, DFL loss
4. **LR schedule**: warmup, cosine decay rate, initial LR
5. **Training duration**: epochs vs patience balance
6. **Small object focus**: multi-scale, higher aug for small targets

## Status Dashboard

After each experiment, update `status.md` with:
- Session info (start time, branch, total experiments)
- Current best CDS and comparison to baseline
- Quality gate progress (precision toward 0.90, recall toward 0.85)
- Stop condition counters
- Last 3 experiments summary
- What's working / what's not
- Next planned experiment

The user can check progress anytime by reading status.md from another terminal.

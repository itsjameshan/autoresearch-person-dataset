"""
dataset_inspector.py - lightweight dataset quality gate for AutoResearch v2.

The inspector runs before training and emits:
  - reports/dataset_report.json
  - reports/dataset_report.md
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SplitStats:
    split: str
    image_dir: str
    label_dir: str
    images: int = 0
    labels_found: int = 0
    labels_missing: int = 0
    labels_empty: int = 0
    malformed_label_lines: int = 0
    class_id_oob: int = 0
    bbox_invalid: int = 0

    def as_dict(self) -> dict[str, Any]:
        d = {
            "split": self.split,
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "images": self.images,
            "labels_found": self.labels_found,
            "labels_missing": self.labels_missing,
            "labels_empty": self.labels_empty,
            "malformed_label_lines": self.malformed_label_lines,
            "class_id_oob": self.class_id_oob,
            "bbox_invalid": self.bbox_invalid,
        }
        d["missing_label_ratio"] = (
            round(self.labels_missing / self.images, 4) if self.images else 0.0
        )
        return d


def _safe_yaml_load(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is not available")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("data yaml root is not an object")
    return data


def _resolve_path(raw: Any, base: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        return base
    p = Path(text)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _infer_label_dir(image_dir: Path, split: str) -> Path:
    parts = list(image_dir.parts)
    lowered = [p.lower() for p in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return (image_dir.parent.parent / "labels" / split).resolve()


def _iter_images(image_dir: Path) -> list[Path]:
    out: list[Path] = []
    if not image_dir.exists():
        return out
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def _check_label_file(path: Path, num_classes: int) -> tuple[int, int, int]:
    malformed = 0
    class_oob = 0
    bbox_invalid = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 1, 0, 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = line.split()
        if len(cols) != 5:
            malformed += 1
            continue
        try:
            cid = int(float(cols[0]))
            x, y, w, h = [float(v) for v in cols[1:]]
        except Exception:
            malformed += 1
            continue

        if cid < 0 or cid >= max(num_classes, 1):
            class_oob += 1

        # YOLO normalized xywh
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            bbox_invalid += 1
    return malformed, class_oob, bbox_invalid


def inspect_dataset(data_yaml_path: str, reports_dir: str = "reports") -> dict[str, Any]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_yaml = Path(data_yaml_path).resolve()
    reports = Path(reports_dir).resolve()
    reports.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at": now,
        "data_yaml": str(data_yaml),
        "blocking": False,
        "severe_issues": [],
        "warnings": [],
        "splits": {},
        "summary": {},
    }

    if not data_yaml.exists():
        report["blocking"] = True
        report["severe_issues"].append(f"data yaml not found: {data_yaml}")
        return _write_reports(report, reports)

    try:
        cfg = _safe_yaml_load(data_yaml)
    except Exception as e:
        report["blocking"] = True
        report["severe_issues"].append(f"failed to parse data yaml: {e}")
        return _write_reports(report, reports)

    num_classes = int(cfg.get("nc", 1) or 1)
    yaml_parent = data_yaml.parent
    path_from_yaml = cfg.get("path", ".")
    if not os.path.isabs(path_from_yaml):
        base = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / path_from_yaml
    else:
        base = Path(path_from_yaml)

    split_names = ["train", "val", "test"]
    split_stats: dict[str, SplitStats] = {}
    seen_names: dict[str, set[str]] = {}

    for split in split_names:
        raw = cfg.get(split)
        if raw is None:
            continue
        image_dir = _resolve_path(raw, base)
        label_dir = _infer_label_dir(image_dir, split)
        stats = SplitStats(
            split=split,
            image_dir=str(image_dir),
            label_dir=str(label_dir),
        )
        split_stats[split] = stats

        if not image_dir.exists():
            report["severe_issues"].append(f"{split}: image directory missing: {image_dir}")
            continue
        if not label_dir.exists():
            report["severe_issues"].append(f"{split}: label directory missing: {label_dir}")

        images = _iter_images(image_dir)
        stats.images = len(images)
        seen_names[split] = {p.name.lower() for p in images}

        if split in ("train", "val") and stats.images == 0:
            report["severe_issues"].append(f"{split}: no images found")

        for img in images:
            label_path = label_dir / f"{img.stem}.txt"
            if not label_path.exists():
                stats.labels_missing += 1
                continue
            stats.labels_found += 1
            try:
                if label_path.stat().st_size == 0:
                    stats.labels_empty += 1
            except Exception:
                pass
            malformed, class_oob, bbox_invalid = _check_label_file(label_path, num_classes)
            stats.malformed_label_lines += malformed
            stats.class_id_oob += class_oob
            stats.bbox_invalid += bbox_invalid

    if "train" in seen_names and "val" in seen_names:
        dup = seen_names["train"].intersection(seen_names["val"])
        if dup:
            # Filename overlap is a useful signal, but often too noisy to hard-block.
            report["warnings"].append(f"train/val duplicate filenames: {len(dup)}")

    total_images = 0
    total_missing = 0
    total_malformed = 0
    total_bbox_invalid = 0
    total_class_oob = 0

    for split, stats in split_stats.items():
        data = stats.as_dict()
        report["splits"][split] = data
        total_images += stats.images
        total_missing += stats.labels_missing
        total_malformed += stats.malformed_label_lines
        total_bbox_invalid += stats.bbox_invalid
        total_class_oob += stats.class_id_oob

        if split in ("train", "val") and stats.images > 0:
            miss_ratio = stats.labels_missing / stats.images
            if miss_ratio > 0.2:
                report["severe_issues"].append(
                    f"{split}: missing labels ratio too high ({miss_ratio:.1%})"
                )
            elif miss_ratio > 0:
                report["warnings"].append(
                    f"{split}: missing labels ratio {miss_ratio:.1%}"
                )

    # Label format issues: hard-block only when volume suggests systemic corruption.
    severe_row_threshold = max(20, int(total_images * 0.01))
    if total_malformed > 0:
        msg = f"malformed label lines: {total_malformed}"
        if total_malformed >= severe_row_threshold:
            report["severe_issues"].append(msg)
        else:
            report["warnings"].append(msg)
    if total_bbox_invalid > 0:
        msg = f"invalid bbox rows: {total_bbox_invalid}"
        if total_bbox_invalid >= severe_row_threshold:
            report["severe_issues"].append(msg)
        else:
            report["warnings"].append(msg)
    if total_class_oob > 0:
        msg = f"class id out of bounds rows: {total_class_oob}"
        if total_class_oob >= severe_row_threshold:
            report["severe_issues"].append(msg)
        else:
            report["warnings"].append(msg)

    report["summary"] = {
        "total_images": total_images,
        "total_missing_labels": total_missing,
        "total_malformed_label_lines": total_malformed,
        "total_invalid_bbox_rows": total_bbox_invalid,
        "total_class_id_oob_rows": total_class_oob,
    }
    report["blocking"] = len(report["severe_issues"]) > 0
    return _write_reports(report, reports)


def _write_reports(report: dict[str, Any], reports_dir: Path) -> dict[str, Any]:
    json_path = reports_dir / "dataset_report.json"
    md_path = reports_dir / "dataset_report.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Dataset Report",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- data_yaml: `{report.get('data_yaml', '')}`",
        f"- blocking: **{report.get('blocking', False)}**",
        "",
        "## Summary",
    ]
    for k, v in (report.get("summary") or {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Severe Issues")
    severe = report.get("severe_issues") or []
    if severe:
        for item in severe:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Warnings")
    warns = report.get("warnings") or []
    if warns:
        for item in warns:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Split Details")
    for split, info in (report.get("splits") or {}).items():
        lines.append(f"### {split}")
        for k, v in info.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    return report


# Experiment Suggestions

## Hardware Advantage (RTX 5070 12GB + 64GB RAM)
- Can use full imgsz=1280 with batch=16 (vs Mac M1 limited to imgsz=320 batch=8 CPU)
- cache="ram" — entire dataset in memory, no I/O bottleneck
- AMP FP16 on CUDA is fast and stable

## Next Suggestions
1. Run baseline first (yolov8s, imgsz=1280, batch=16)
2. After baseline, try yolo12s (expected biggest improvement)
3. Increase copy_paste 0.1→0.3 (more small object augmentation)
4. Try yolo12l with batch=8 (12GB VRAM might support it)
5. Increase batch to 24 if yolov8s baseline uses <8GB VRAM

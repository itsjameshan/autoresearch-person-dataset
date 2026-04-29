"""
记录实验结果到results.tsv
使用方式: python record_result.py <实验名称> <模型> <mAP50> <precision> <recall> <small_obj_recall> <counting_mae> <备注>
"""

import sys
import os
from datetime import datetime

RESULTS_FILE = "results.tsv"

def main():
    if len(sys.argv) < 8:
        print("使用方式:")
        print("  python record_result.py <实验名称> <模型> <mAP50> <precision> <recall> <small_obj_recall> <counting_mae> [备注]")
        print("")
        print("示例:")
        print('  python record_result.py "exp01_yolo12m_baseline" yolo12m 0.72 0.75 0.78 0.68 2.1 "yolo12m baseline - 10%提升"')
        sys.exit(1)
    
    exp_name = sys.argv[1]
    model = sys.argv[2]
    map50 = sys.argv[3]
    precision = sys.argv[4]
    recall = sys.argv[5]
    small_recall = sys.argv[6]
    counting_mae = sys.argv[7]
    note = sys.argv[8] if len(sys.argv) > 8 else ""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 检查文件是否存在，如果不存在则创建表头
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("时间\t实验名称\t模型\tmAP50\tprecision\trecall\tsmall_obj_recall\tcounting_mae\t备注\n")
    
    # 追加结果
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp}\t{exp_name}\t{model}\t{map50}\t{precision}\t{recall}\t{small_recall}\t{counting_mae}\t{note}\n")
    
    print(f"结果已记录到 {RESULTS_FILE}")
    print(f"实验: {exp_name}")
    print(f"mAP50: {map50}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == "__main__":
    main()
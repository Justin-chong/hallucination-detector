"""
幻觉检测评估脚本
==================

对比 detector 输出与 ground_truth.json，输出：
  - 二分类指标：precision / recall / F1 / accuracy
  - 分类层面：混淆矩阵（5 类）
  - 漏检列表（false negative）与误报列表（false positive）
  - Markdown 评估报告

用法：
  python evaluate.py
  python evaluate.py --predictions reports/result.json --ground-truth data/task4_ground_truth.json
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict


def load_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_binary_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """二分类指标（是否幻觉）"""
    gt_map = {g["id"]: g for g in ground_truth}
    pred_map = {p["id"]: p for p in predictions}

    tp = fp = tn = fn = 0
    misclassified = []

    for id_, gt in gt_map.items():
        pred = pred_map.get(id_)
        if not pred:
            continue
        gt_pos = gt["is_hallucination"]
        pred_pos = pred["is_hallucination"]

        if gt_pos and pred_pos:
            tp += 1
        elif not gt_pos and not pred_pos:
            tn += 1
        elif gt_pos and not pred_pos:
            fn += 1
            misclassified.append({"id": id_, "type": "FN", "gt_type": gt.get("hallucination_type")})
        elif not gt_pos and pred_pos:
            fp += 1
            misclassified.append({"id": id_, "type": "FP", "pred_type": pred.get("hallucination_type")})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "support_pos": tp + fn,
        "support_neg": tn + fp,
        "misclassified": misclassified,
    }


def compute_type_confusion(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """分类层混淆矩阵（按类型）"""
    gt_map = {g["id"]: g for g in ground_truth}
    pred_map = {p["id"]: p for p in predictions}

    matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    correct = 0
    total = 0

    for id_, gt in gt_map.items():
        pred = pred_map.get(id_)
        if not pred:
            continue
        gt_type = gt.get("hallucination_type") or "正常"
        pred_type = pred.get("hallucination_type") or "正常"
        matrix[(gt_type, pred_type)] += 1
        total += 1
        if gt_type == pred_type:
            correct += 1

    return {
        "matrix": dict(matrix),
        "correct": correct,
        "total": total,
        "type_accuracy": correct / total if total > 0 else 0,
    }


def format_binary_report(metrics: Dict) -> List[str]:
    """二分类指标报告"""
    lines = [
        "## 二分类指标（是否幻觉）\n\n",
        "| 指标 | 值 |\n|---|---|\n",
        f"| TP (真阳性) | {metrics['tp']} |\n",
        f"| FP (误报) | {metrics['fp']} |\n",
        f"| TN (真阴性) | {metrics['tn']} |\n",
        f"| FN (漏检) | {metrics['fn']} |\n",
        f"| **Precision** | **{metrics['precision']:.3f}** |\n",
        f"| **Recall** | **{metrics['recall']:.3f}** |\n",
        f"| **F1** | **{metrics['f1']:.3f}** |\n",
        f"| **Accuracy** | **{metrics['accuracy']:.3f}** |\n",
        f"| 正样本数 | {metrics['support_pos']} |\n",
        f"| 负样本数 | {metrics['support_neg']} |\n\n",
    ]
    return lines


def format_confusion_matrix(matrix_data: Dict, my_types: List[str], gt_types: List[str]) -> List[str]:
    """分类混淆矩阵"""
    matrix = matrix_data["matrix"]
    all_types = sorted(set(my_types) | set(gt_types))

    lines = ["## 分类混淆矩阵\n\n"]
    lines.append(f"分类准确率：**{matrix_data['type_accuracy']:.3f}** ({matrix_data['correct']}/{matrix_data['total']})\n\n")
    lines.append(f"> 行 = 真实标签（ground_truth），列 = 预测标签（detector）\n\n")

    # 表头
    lines.append("| 真 \\ 预 |")
    for t in all_types:
        lines.append(f" {t} |")
    lines.append("\n")
    lines.append("|" + "---|" * (len(all_types) + 1) + "\n")

    for gt in all_types:
        lines.append(f"| **{gt}** |")
        for pred in all_types:
            count = matrix.get((gt, pred), 0)
            if count > 0:
                lines.append(f" {count} |")
            else:
                lines.append(f" · |")
        lines.append("\n")
    lines.append("\n")
    return lines


def format_misclassified(metrics: Dict) -> List[str]:
    """漏检/误报 case 列表"""
    mis = metrics["misclassified"]
    if not mis:
        return ["## 误判分析\n\n**所有 case 均正确，无漏检无误报。**\n\n"]

    lines = ["## 误判分析\n\n"]
    fns = [m for m in mis if m["type"] == "FN"]
    fps = [m for m in mis if m["type"] == "FP"]
    lines.append(f"### 漏检（FN）：{len(fns)} 条\n\n")
    for m in fns:
        lines.append(f"- **{m['id']}** — 真实: {m['gt_type']}\n")
    lines.append(f"\n### 误报（FP）：{len(fps)} 条\n\n")
    for m in fps:
        lines.append(f"- **{m['id']}** — 预测: {m['pred_type']}\n")
    lines.append("\n")
    return lines


def main():
    parser = argparse.ArgumentParser(description="幻觉检测评估")
    parser.add_argument(
        "--predictions", "-p", default="reports/result.json",
        help="预测结果 JSON 路径"
    )
    parser.add_argument(
        "--ground-truth", "-g", default="data/task4_ground_truth.json",
        help="ground truth JSON 路径"
    )
    parser.add_argument(
        "--output-dir", "-o", default="reports",
        help="评估报告输出目录"
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    pred_path = Path(args.predictions) if Path(args.predictions).is_absolute() else base / args.predictions
    gt_path = Path(args.ground_truth) if Path(args.ground_truth).is_absolute() else base / args.ground_truth
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else base / args.output_dir

    predictions = load_json(pred_path)
    ground_truth = load_json(gt_path)

    metrics = compute_binary_metrics(predictions, ground_truth)
    matrix_data = compute_type_confusion(predictions, ground_truth)

    # 类型映射（detector 5 类 + 正常；ground_truth 8 类 + 正常）
    my_types = sorted({p.get("hallucination_type") or "正常" for p in predictions})
    gt_types = sorted({g.get("hallucination_type") or "正常" for g in ground_truth})

    # 控制台输出
    print("\n" + "=" * 60)
    print("幻觉检测评估")
    print("=" * 60)
    print(f"样本数: {len(ground_truth)}")
    print(f"  真实幻觉: {metrics['support_pos']}")
    print(f"  真实正常: {metrics['support_neg']}")
    print()
    print(f"TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    print()
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print()
    print(f"分类准确率: {matrix_data['type_accuracy']:.3f}")
    print()
    if metrics["misclassified"]:
        print("误判 case:")
        for m in metrics["misclassified"]:
            print(f"  {m}")
    else:
        print("✓ 无误判")

    # Markdown 报告
    output_dir.mkdir(parents=True, exist_ok=True)
    md = ["# 幻觉检测评估报告\n\n"]
    md.append(f"> 评估对象：`{pred_path}` vs `{gt_path}`\n\n")
    md.extend(format_binary_report(metrics))
    md.extend(format_confusion_matrix(matrix_data, my_types, gt_types))
    md.extend(format_misclassified(metrics))

    md_path = output_dir / "evaluation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"\n评估报告已写入: {md_path}")

    # JSON 报告（机器可读）
    eval_json = {
        "binary_metrics": {
            k: v for k, v in metrics.items() if k != "misclassified"
        },
        "type_accuracy": matrix_data["type_accuracy"],
        "type_confusion_matrix": {f"{k[0]}|{k[1]}": v for k, v in matrix_data["matrix"].items()},
        "misclassified_ids": [m["id"] for m in metrics["misclassified"]],
    }
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=2)
    print(f"评估结果 JSON: {eval_path}")


if __name__ == "__main__":
    main()
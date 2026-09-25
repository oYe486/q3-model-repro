"""Reproduce the question-3 modality comparison from frozen validation outputs.

The 728 labeled attachment-2 validation rows supply all quantitative results.
The 20 unlabeled attachment-4 rows supply two illustrations only. The optional
window rescan uses the frozen checkpoint and does not train or tune anything.
"""

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "modality_roles"
NAMES = ("text", "audio", "vision")
LABELS = {"text": "Text", "audio": "Audio", "vision": "Vision"}
COLORS = {"text": "#2563A6", "audio": "#E69F36", "vision": "#559C75"}
CLASSES = {0: "Negative", 1: "Neutral", 2: "Positive"}
CASE_IDS = ("09", "19")


def read_inputs():
    with (ROOT / "outputs/validation/valid_predictions.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metrics = json.loads((ROOT / "outputs/validation/valid_thresholds.json")
                         .read_text(encoding="utf-8"))
    attachment = [json.loads(line) for line in
                  (ROOT / "outputs/attachment4/predictions_20.jsonl")
                  .read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 728 or len(attachment) != 20:
        raise AssertionError("Expected 728 validation and 20 attachment-4 rows")
    if len({int(row["sample_index"]) for row in rows}) != len(rows):
        raise AssertionError("Validation sample indices are not unique")
    if len({row["sample_id"] for row in attachment}) != len(attachment):
        raise AssertionError("Attachment-4 sample IDs are not unique")
    if {row["sample_id"] for row in attachment} != {
            f"{i:02d}" for i in range(1, 21)}:
        raise AssertionError("Attachment-4 IDs are incomplete")
    return rows, metrics, attachment


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def pct(n, d):
    return 100 * n / d if d else float("nan")


def check_and_summarize(rows, metrics, attachment):
    n = len(rows)
    accuracy = mean([int(r["predicted_class"] == r["true_class"])
                     for r in rows])
    mae = mean([abs(float(r["predicted_score"]) - float(r["true_score"]))
                for r in rows])
    if not math.isclose(accuracy, metrics["accuracy"], abs_tol=1e-10):
        raise AssertionError("Accuracy disagrees with frozen summary")
    if not math.isclose(mae, metrics["mae"], abs_tol=1e-10):
        raise AssertionError("MAE disagrees with frozen summary")
    main = Counter(r["main_modality"] for r in rows)
    gate = Counter(r["fusion_main_modality"] for r in rows)
    if dict(main) != metrics["coalition_audit"]["main_counts"]:
        raise AssertionError("Main-modality counts disagree")
    if dict(gate) != metrics["gate_main_counts"]:
        raise AssertionError("Gate-modality counts disagree")
    weights = {m: mean([float(r[f"weight_{m}"]) for r in rows]) for m in NAMES}
    for i, m in enumerate(NAMES):
        if not math.isclose(weights[m], metrics["gate_mean_weights"][i],
                            abs_tol=1e-7):
            raise AssertionError("Gate means disagree")
    shares = {m: mean([float(r[f"positive_share_{m}"]) for r in rows])
              for m in NAMES}
    signed = {m: mean([float(r[f"shapley_support_{m}"]) for r in rows])
              for m in NAMES}
    by_class = {}
    for cls in CLASSES:
        group = [r for r in rows if int(r["predicted_class"]) == cls]
        by_class[str(cls)] = {
            "n": len(group),
            "accuracy": mean([int(r["predicted_class"] == r["true_class"])
                              for r in group]),
            "main_counts": dict(Counter(r["main_modality"] for r in group)),
            "positive_share_means": {
                m: mean([float(r[f"positive_share_{m}"]) for r in group])
                for m in NAMES},
        }
    cross = {gate_m: {main_m: sum(r["fusion_main_modality"] == gate_m and
                                  r["main_modality"] == main_m for r in rows)
                      for main_m in NAMES} for gate_m in NAMES}
    removal = {}
    for m in NAMES:
        group = [r for r in rows if r[f"label_error_increment_without_{m}"]
                 and r[f"label_error_increment_without_{m}"].lower() != "nan"]
        increments = [float(r[f"label_error_increment_without_{m}"])
                      for r in group]
        flips = [r[f"class_changed_without_{m}"] == "True" for r in group]
        if any(f"predicted_class_without_{m}" not in r for r in group):
            raise AssertionError("Rerun frozen validation to persist removal classes")
        if any((r[f"predicted_class_without_{m}"] != r["predicted_class"])
               != (r[f"class_changed_without_{m}"] == "True")
               for r in group):
            raise AssertionError(f"Removal class/flip mismatch for {m}")
        correct_to_wrong = sum(
            r["predicted_class"] == r["true_class"] and
            r[f"predicted_class_without_{m}"] != r["true_class"]
            for r in group)
        wrong_to_correct = sum(
            r["predicted_class"] != r["true_class"] and
            r[f"predicted_class_without_{m}"] == r["true_class"]
            for r in group)
        removal[m] = {
            "n_available": len(group),
            "mean_label_error_increment": mean(increments),
            "flip_n": sum(flips), "flip_rate": mean(flips),
            "positive_error_n": sum(x > 0 for x in increments),
            "negative_error_n": sum(x < 0 for x in increments),
            "correct_to_wrong": correct_to_wrong,
            "wrong_to_correct": wrong_to_correct,
        }
    selected_changed = selected_correct_to_wrong = selected_wrong_to_correct = 0
    for row in rows:
        after = row[f"predicted_class_without_{row['main_modality']}"]
        selected_changed += after != row["predicted_class"]
        selected_correct_to_wrong += (row["predicted_class"] == row["true_class"]
                                      and after != row["true_class"])
        selected_wrong_to_correct += (row["predicted_class"] != row["true_class"]
                                      and after == row["true_class"])
    audit = metrics["coalition_audit"]
    if (selected_changed != audit["n_selected_class_flip"] or
            selected_correct_to_wrong != audit["n_selected_correct_to_wrong"] or
            selected_wrong_to_correct != audit["n_selected_wrong_to_correct"]):
        raise AssertionError("Selected-modality transitions disagree with summary")
    attachment_main = Counter(r["main_modality"] for r in attachment)
    summary = {
        "source": "attachment_2_aligned_50_validation_only",
        "n_validation": n, "accuracy": accuracy, "mae": mae,
        "macro_f1": metrics["macro_f1"],
        "pearson_correlation": metrics["correlation"],
        "gate_mean_weights": weights, "positive_share_means": shares,
        "signed_support_means": signed,
        "gate_main_counts": dict(gate), "explanation_main_counts": dict(main),
        "main_gate_agreement": sum(cross[m][m] for m in NAMES) / n,
        "main_gate_cross": cross, "by_predicted_class": by_class,
        "removal": removal,
        "attachment4_n": len(attachment),
        "attachment4_main_counts": dict(attachment_main),
        "attachment4_used_for_quantitative_validation": False,
        "offline_proxy_clear_n": metrics["coalition_audit"]["n_clear_label_benefit"],
        "explanation_proxy_agreement": metrics["coalition_audit"]
        ["match_label_benefit_oracle_clear"],
        "always_text_proxy_agreement": metrics["coalition_audit"]
        ["always_text_match_label_benefit_oracle_clear"],
    }
    if not math.isclose(summary["main_gate_agreement"],
                        metrics["coalition_audit"]["fusion_gate_agreement"],
                        abs_tol=1e-10):
        raise AssertionError("Gate/explanation agreement disagrees")
    return summary


def svg_start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="40" y="37" font-family="Arial,sans-serif" '
            f'font-size="20" font-weight="bold" fill="#172334">'
            f'{html.escape(title)}</text>']


def txt(x, y, value, size=13, anchor="start", color="#283446",
        weight="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="Arial,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{html.escape(str(value))}</text>')


def rect(x, y, w, h, color, opacity=1):
    return (f'<rect x="{x}" y="{y}" width="{max(w, 0):.2f}" '
            f'height="{max(h, 0):.2f}" fill="{color}" opacity="{opacity}"/>')


def finish(parts, path):
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def legend(parts, y, x=40):
    for m in NAMES:
        parts.append(rect(x, y - 11, 14, 14, COLORS[m]))
        parts.append(txt(x + 21, y, LABELS[m], 13))
        x += 110


def chart_class_shares(summary):
    parts = svg_start(850, 395, "Positive decision-support share by predicted class")
    legend(parts, 73)
    parts.append(txt(60, 110, "Share (%)", 12))
    for tick in range(0, 101, 25):
        y = 310 - tick * 1.7
        parts.append(f'<line x1="100" y1="{y}" x2="800" y2="{y}" '
                     'stroke="#E3E8EE"/>')
        parts.append(txt(88, y + 4, tick, 11, "end"))
    for i, cls in enumerate(CLASSES):
        item = summary["by_predicted_class"][str(cls)]
        x = 160 + i * 230
        current = 310
        for m in NAMES:
            value = item["positive_share_means"][m]
            height = 170 * value
            current -= height
            parts.append(rect(x, current, 110, height, COLORS[m]))
            if value >= .08:
                parts.append(txt(x + 55, current + height / 2 + 4,
                                 f"{value * 100:.1f}%", 12, "middle", "white",
                                 "bold"))
        parts.append(txt(x + 55, 336, CLASSES[cls], 14, "middle", weight="bold"))
        parts.append(txt(x + 55, 356,
                         f"n={item['n']}, accuracy={item['accuracy']:.1%}",
                         12, "middle"))
    finish(parts, OUT / "fig1_support_share_by_class.svg")


def chart_cross(summary):
    parts = svg_start(690, 475, "Fusion gate versus explanation main modality")
    parts.append(txt(345, 79, "Explanation main modality", 14, "middle"))
    for j, m in enumerate(NAMES):
        parts.append(txt(270 + j * 125, 111, LABELS[m], 13, "middle"))
    for i, gate in enumerate(NAMES):
        parts.append(txt(160, 160 + i * 90, LABELS[gate], 13, "end"))
        for j, main in enumerate(NAMES):
            value = summary["main_gate_cross"][gate][main]
            opacity = 0.12 + .82 * value / 600
            parts.append(rect(205 + j * 125, 123 + i * 90, 120, 78,
                              COLORS[main], opacity))
            parts.append(txt(265 + j * 125, 166 + i * 90, value, 19,
                             "middle", "#102034", "bold"))
    parts.append(txt(30, 428, "Rows: gate choice; columns: predicted-class Shapley choice.", 12))
    parts.append(txt(30, 449,
                     f"Agreement: {summary['main_gate_agreement']:.1%} of 728 validation samples.",
                     12))
    finish(parts, OUT / "fig2_gate_vs_explanation.svg")


def chart_removal(summary):
    parts = svg_start(875, 450, "Effect of removing each modality on validation predictions")
    legend(parts, 74)
    parts.append(txt(65, 114, "Mean change in absolute label error", 13,
                     weight="bold"))
    parts.append(txt(490, 114, "Predicted-class flip rate", 13,
                     weight="bold"))
    for i, m in enumerate(NAMES):
        y = 153 + i * 88
        r = summary["removal"][m]
        parts.append(txt(30, y + 22, LABELS[m], 13))
        # Zero is fixed at x=260, with enough room for negative deltas.
        parts.append(f'<line x1="260" y1="140" x2="260" y2="410" '
                     'stroke="#8894A0"/>')
        delta = r["mean_label_error_increment"]
        parts.append(rect(260 if delta >= 0 else 260 + 650 * delta,
                          y, abs(650 * delta), 31, COLORS[m]))
        parts.append(txt(270 + max(0, 650 * delta), y + 22,
                         f"{delta:+.3f}", 12))
        parts.append(rect(510, y, 300 * r["flip_rate"], 31,
                          COLORS[m]))
        parts.append(txt(520 + 300 * r["flip_rate"], y + 22,
                         f"{r['flip_rate']:.1%} ({r['flip_n']}/{r['n_available']})", 12))
    parts.append(txt(30, 428,
                     "Positive error change means removal worsened MAE; flip rate does not imply a correctness gain.",
                     12))
    finish(parts, OUT / "fig3_modality_removal.svg")


def recompute_windows(attachment, metrics):
    import torch
    from infer_attachment4 import _batch, _score, _top_windows
    from trains.singleTask.model.DLF import DLF
    from utils.attachment4_data import load_aligned_sample

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DLF.from_final_checkpoint(root=ROOT, device=device)
    records = {r["sample_id"]: r for r in attachment}
    result = {}
    with torch.no_grad():
        for sample_id in CASE_IDS:
            record = records[sample_id]
            sample = load_aligned_sample(
                ROOT.parent / "datasets/附件4-可解释专项视频样本与特征文件"
                / "对齐版本" / f"{sample_id}.pkl")
            base = _batch(sample, device)
            score = float(_score(model, base)[0])
            if not math.isclose(score, record["predicted_score"], abs_tol=1e-4):
                raise AssertionError(f"Frozen score changed for sample {sample_id}")
            windows = _top_windows(
                model, base, score, record["predicted_class"],
                metrics["threshold_negative_neutral"],
                metrics["threshold_neutral_positive"], 3, 100, 16)
            result[sample_id] = {
                "sample_id": sample_id,
                "main_modality": record["main_modality"],
                "predicted_score": score,
                "window_width": 3,
                "window_selection": "nonoverlapping observed aligned positions",
                "by_modality": {m: sorted(windows[m], key=lambda x: x["start"])
                                for m in NAMES},
            }
            top = max(windows[record["main_modality"]],
                      key=lambda x: x["decision_support"])
            saved = record["primary_evidence"]
            if (top["start"] != saved["start"] or
                    top["end_exclusive"] != saved["end_exclusive"] or
                    not math.isclose(top["decision_support"],
                                     saved["decision_support"], abs_tol=1e-4)):
                raise AssertionError(f"Primary window changed for {sample_id}")
    (OUT / "local_windows_09_19.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def chart_local(windows):
    parts = svg_start(920, 560, "Local occlusion importance in two attachment-4 cases")
    for k, sample_id in enumerate(CASE_IDS):
        record = windows[sample_id]
        m = record["main_modality"]
        values = record["by_modality"][m]
        max_abs = max(abs(item["decision_support"]) for item in values)
        max_abs = max(max_abs, .01)
        left, width = 85 + 450 * k, 350
        baseline = 290
        parts.append(txt(left, 90,
                         f"Sample {sample_id}: {LABELS[m]} main modality",
                         15, weight="bold"))
        parts.append(txt(left, 116,
                         f"Score {record['predicted_score']:+.3f}; "
                         f"{len(values)} observed windows", 12))
        parts.append(f'<line x1="{left}" y1="{baseline}" '
                     f'x2="{left + width}" y2="{baseline}" stroke="#8A949F"/>')
        parts.append(txt(left - 7, baseline + 4, "0", 11, "end"))
        parts.append(txt(left - 7, baseline - 105 + 4,
                         f"+{max_abs:.3f}", 11, "end"))
        parts.append(txt(left - 7, baseline + 105 + 4,
                         f"-{max_abs:.3f}", 11, "end"))
        coords = []
        min_start = min(item["start"] for item in values)
        max_start = max(item["start"] for item in values)
        for item in values:
            x = left + (item["start"] - min_start) / max(1, max_start - min_start) * width
            y = baseline - item["decision_support"] / max_abs * 105
            coords.append((x, y, item))
        parts.append('<polyline points="' + " ".join(
            f"{x:.1f},{y:.1f}" for x, y, _ in coords) +
            f'" fill="none" stroke="{COLORS[m]}" stroke-width="2.5"/>')
        for x, y, _ in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
                         f'fill="{COLORS[m]}"/>')
        top = max(coords, key=lambda triple: triple[2]["decision_support"])
        parts.append(f'<circle cx="{top[0]:.1f}" cy="{top[1]:.1f}" r="7" '
                     'fill="none" stroke="#C7313A" stroke-width="2"/>')
        parts.append(txt(left, 425,
                         f"Top window [{top[2]['start']}, "
                         f"{top[2]['end_exclusive']}), "
                         f"support {top[2]['decision_support']:+.4f}", 12))
        parts.append(txt(left, 448,
                         f"Window starts: {min_start} to {max_start}", 12))
        parts.append(txt(left, 470, "X: aligned feature position (window start)", 12))
        parts.append(txt(left, 492, "Y: original-class margin drop on occlusion", 12))
    parts.append(txt(45, 535,
                     "Audio time for sample 19 is a candidate mapping, not an authenticated feature timestamp.",
                     12))
    finish(parts, OUT / "fig4_local_importance_cases.svg")


def report(summary, attachment, windows):
    cases = {r["sample_id"]: r for r in attachment}
    lines = [
        "# 问题三 三模态作用差异对比分析",
        "",
        "本分析以附件2的 aligned_50 验证集728条样本为定量依据。模型、阈值及解释规则保持冻结；附件4的20条无标签样本只用于案例展示。三分类阈值与本报告的分类指标来自同一验证集，不属于独立测试集估计。",
        "",
        "## 定义与整体差异",
        "",
        "融合门控权重表示模型如何路由三模态；主要参考模态由当前预测类别决策边际的八组合精确 Shapley 值确定。正向支持份额为 `max(φ_m,0)/Σ_j max(φ_j,0)`；有符号的 `φ_m` 另行保留，负值不能删除或称为负权重。局部重要性是遮挡有效特征窗口后，原预测类别决策边际的下降。以上量均描述冻结模型的行为，不是现实因果效应。",
        "",
        "| 模态 | 平均门控权重 | 平均正向支持份额 | 主要参考模态数量 | 平均有符号支持 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in NAMES:
        lines.append(f"| {LABELS[m]} | {summary['gate_mean_weights'][m]:.3f} | "
                     f"{summary['positive_share_means'][m]:.3f} | "
                     f"{summary['explanation_main_counts'][m]} | "
                     f"{summary['signed_support_means'][m]:+.3f} |")
    lines += [
        "",
        f"门控主模态与解释主模态一致率为 {summary['main_gate_agreement']:.1%}。图1展示分组正向支持份额，图2展示两个主模态定义的交叉分布。",
        "",
        "![按预测类别划分的正向支持份额](../outputs/modality_roles/fig1_support_share_by_class.svg)",
        "",
        "![门控与解释主模态交叉分布](../outputs/modality_roles/fig2_gate_vs_explanation.svg)",
        "",
        "## 按预测类别比较",
        "",
        "| 预测类别 | 样本数 | 该组预测正确率 | 文本主导 | 语音主导 | 视觉主导 | 文本/语音/视觉平均正向份额 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cls in CLASSES:
        item = summary["by_predicted_class"][str(cls)]
        counts = item["main_counts"]
        shares = item["positive_share_means"]
        lines.append(f"| {CLASSES[cls]} | {item['n']} | {item['accuracy']:.1%} | "
                     f"{counts.get('text', 0)} | {counts.get('audio', 0)} | "
                     f"{counts.get('vision', 0)} | "
                     + "/".join(f"{shares[m]:.3f}" for m in NAMES) + " |")
    lines += [
        "",
        "在现有验证记录中，负向和正向预测的主要参考模态大多为文本；中性预测则更多由语音或视觉取得最高决策支持。这是按**模型预测类别**分组的描述，不能直接推断真实中性情感天然更依赖语音或视觉。中性决策边际的定义也与两端类别不同，应结合各组正确率和误判记录解读。",
        "",
        "## 移除模态后的预测变化",
        "",
        "`Δ绝对误差 = |移除后预测强度−真实强度| − |完整模型预测强度−真实强度|`。正值表示移除该模态后标签误差增加；翻类只表示预测类别变化，不表示预测改善。仅对该模态有效的样本统计。",
        "",
        "| 被移除模态 | 有效样本数 | 平均Δ绝对误差 | 翻类样本数与比例 | 正确变错误 / 错误变正确 | 误差增加/减少样本数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in NAMES:
        r = summary["removal"][m]
        lines.append(f"| {LABELS[m]} | {r['n_available']} | "
                     f"{r['mean_label_error_increment']:+.3f} | "
                     f"{r['flip_n']} ({r['flip_rate']:.1%}) | "
                     f"{r['correct_to_wrong']} / {r['wrong_to_correct']} | "
                     f"{r['positive_error_n']}/{r['negative_error_n']} |")
    lines += [
        "",
        "上述分类转移由冻结模型逐一移除各模态后的类别直接计算，不以翻类率代替正确性判断。每种模态的有效样本分母可能不同。",
        "",
        "![移除模态的标签误差变化与翻类比例](../outputs/modality_roles/fig3_modality_removal.svg)",
        "",
        "## 附件4典型解释卡",
        "",
        f"附件4共20条无标签样本，解释主模态为文本{summary['attachment4_main_counts'].get('text', 0)}条、语音{summary['attachment4_main_counts'].get('audio', 0)}条。以下仅展示模型行为，不评价正确率。图4扫描了两条样本的全部有效三位置窗口；图中的横轴是对齐特征索引，不是视频秒数；纵轴是遮挡后原预测类别决策边际的下降。两幅图使用各自的纵轴刻度。",
        "",
    ]
    for sample_id in CASE_IDS:
        r = cases[sample_id]
        location = r["primary_evidence"]["location"]
        m = r["main_modality"]
        parts = [f"{LABELS[q]} {r['modality_signed_support'][q]:+.4f}"
                 for q in NAMES]
        raw = location.get("raw_text_fragment")
        time = (f"{location['media_start_seconds']:.2f}–"
                f"{location['media_end_seconds']:.2f} s"
                if location.get("media_start_seconds") is not None else "未定位")
        evidence = (f"原文“{raw}”（精确字符匹配）" if raw else
                    f"原视频音频时段候选 {time}")
        lines += [
            f"### 样本 {sample_id} {LABELS[m]}主导",
            "",
            f"预测强度 {r['predicted_score']:+.3f}，预测类别 {CLASSES[r['predicted_class']]}；主要参考模态 {LABELS[m]}，融合门控主模态 {LABELS[r['fusion_main_modality']]}。三模态有符号决策支持：{'；'.join(parts)}。",
            "",
            f"主局部窗口为对齐位置 [{r['primary_evidence']['start']}, {r['primary_evidence']['end_exclusive']})，遮挡后模型输出的情感强度为 {r['primary_evidence']['occluded_score']:+.4f}，原预测类别的决策边际下降 {r['primary_evidence']['decision_support']:+.4f}；可回看证据为{evidence}。位置核验状态为 `{location['mapping_status']}`。",
            "",
        ]
    lines += [
        "![两条典型样本的局部遮挡重要性](../outputs/modality_roles/fig4_local_importance_cases.svg)",
        "",
        "样本19的语音秒数是根据音轨识别与原文匹配得到的候选；数据没有原始特征行到媒体时间的权威映射，不应写成已验证的特征提取时间戳。",
        "",
        "## 解释有效性与局限",
        "",
        f"在{summary['offline_proxy_clear_n']}条存在明确标签误差增量胜者的验证样本上，解释主模态与该离线代理的一致率为{summary['explanation_proxy_agreement']:.2%}，低于始终选择文本的{summary['always_text_proxy_agreement']:.2%}。这说明现有结果能量化和复核模型内的决策依据，但不能声称已验证“真实最重要模态”。",
        "",
        "复现：在工程根目录运行 `.\\.venv\\python.exe -m utils.analyze_modality_roles --recompute-windows`。脚本读取已冻结的验证及附件4结果，校验样本数、指标和主模态统计，再输出 JSON、四张 SVG 和本报告。",
    ]
    (ROOT / "docs/modality_role_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-windows", action="store_true",
                        help="Rescan all valid windows of cases 09 and 19")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, metrics, attachment = read_inputs()
    summary = check_and_summarize(rows, metrics, attachment)
    cache = OUT / "local_windows_09_19.json"
    if args.recompute_windows or not cache.exists():
        windows = recompute_windows(attachment, metrics)
    else:
        windows = json.loads(cache.read_text(encoding="utf-8"))
    chart_class_shares(summary)
    chart_cross(summary)
    chart_removal(summary)
    chart_local(windows)
    (OUT / "statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report(summary, attachment, windows)
    print(json.dumps({"n_validation": summary["n_validation"],
                      "figures": 4,
                      "report": str(ROOT / "docs/modality_role_analysis.md")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

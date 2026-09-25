"""Reproduce paper-ready validation diagnostics from the frozen prediction CSV.

This script never trains a model or selects thresholds. It reads the saved
validation predictions and the original validation metadata for case review.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/validation/valid_predictions.csv"
METRICS = ROOT / "outputs/validation/valid_thresholds.json"
FEATURES = ROOT.parent / "datasets/附件2-数据集特征文件/aligned_50.pkl"
OUT = ROOT / "outputs/validation/analysis"
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
CLASSES = ("Negative", "Neutral", "Positive")
COLORS = ("#355C8A", "#D69A45", "#39856D")


def font(size: int, bold: bool = False):
    path = FONT_BOLD_PATH if bold else FONT_PATH
    return ImageFont.truetype(str(path), size)


def text(draw, xy, value, size=22, fill="#233142", bold=False, anchor=None):
    draw.text(xy, str(value), font=font(size, bold), fill=fill, anchor=anchor)


def confusion_plot(cm: np.ndarray):
    im = Image.new("RGB", (1000, 840), "#ffffff")
    d = ImageDraw.Draw(im)
    text(d, (80, 42), "Validation confusion matrix", 34, bold=True)
    text(d, (80, 88), "Rows: reference label   Columns: model prediction   n = 728", 20, "#5D6B79")
    x0, y0, cell = 280, 180, 170
    vmax = max(int(cm.max()), 1)
    for i in range(3):
        text(d, (x0 - 25, y0 + i * cell + cell / 2), CLASSES[i], 22, anchor="rm")
        text(d, (x0 + i * cell + cell / 2, y0 - 20), CLASSES[i], 22, anchor="mb")
        for j in range(3):
            n = int(cm[i, j])
            strength = n / vmax
            base = (235, 243, 249)
            dark = (44, 98, 144)
            rgb = tuple(round(base[k] + (dark[k] - base[k]) * strength) for k in range(3))
            box = (x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - 5, y0 + (i + 1) * cell - 5)
            d.rounded_rectangle(box, radius=12, fill=rgb)
            text(d, (x0 + j * cell + (cell - 5) / 2, y0 + i * cell + 69), n, 37,
                 "#ffffff" if strength > .48 else "#18324C", bold=True, anchor="mm")
            text(d, (x0 + j * cell + (cell - 5) / 2, y0 + i * cell + 111),
                 f"{n / cm[i].sum():.1%} of row", 17,
                 "#ffffff" if strength > .48 else "#40556B", anchor="mm")
    text(d, (80, 750), "Accuracy 63.05%     Macro-F1 61.58%     Neutral F1 46.97%", 22)
    im.save(OUT / "validation_confusion.png")


def scatter_plot(df: pd.DataFrame, lower: float, upper: float):
    im = Image.new("RGB", (1160, 920), "#ffffff")
    d = ImageDraw.Draw(im)
    text(d, (90, 36), "Validation sentiment strength", 34, bold=True)
    text(d, (90, 83), "Each point is one sample; vertical distance from the diagonal is prediction error", 20, "#5D6B79")
    left, top, right, bottom = 125, 155, 1060, 800
    xmin, xmax, ymin, ymax = -3.15, 3.15, -3.15, 3.15
    px = lambda v: left + (float(v) - xmin) / (xmax - xmin) * (right - left)
    py = lambda v: bottom - (float(v) - ymin) / (ymax - ymin) * (bottom - top)
    for tick in range(-3, 4):
        x, y = px(tick), py(tick)
        d.line((x, top, x, bottom), fill="#E6EBEF", width=1)
        d.line((left, y, right, y), fill="#E6EBEF", width=1)
        text(d, (x, bottom + 12), tick, 17, anchor="mt")
        text(d, (left - 13, y), tick, 17, anchor="rm")
    d.line((px(-3), py(-3), px(3), py(3)), fill="#8797A5", width=3)
    for threshold in (lower, upper):
        y = py(threshold)
        d.line((left, y, right, y), fill="#D6A665", width=2)
    # Draw correctly classified observations first, then errors on top.
    for correct in (True, False):
        sub = df[df.correct == correct]
        for row in sub.itertuples():
            x, y = px(row.true_score), py(row.predicted_score)
            radius = 4 if correct else 5
            color = COLORS[int(row.true_class)] if correct else "#BC4955"
            d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    d.rectangle((left, top, right, bottom), outline="#8EA0AF", width=2)
    text(d, ((left + right) / 2, 850), "Reference sentiment strength", 23, anchor="mm")
    text(d, (90, 122), "Predicted strength", 19)
    d.ellipse((790, 110, 803, 123), fill="#BC4955")
    text(d, (815, 116), "Incorrect polarity", 17, anchor="lm")
    im.save(OUT / "validation_strength_scatter.png")


def near_threshold_fraction(df, lower, upper, width=.1):
    near = (np.abs(df.predicted_score - lower) <= width) | (np.abs(df.predicted_score - upper) <= width)
    return int(near.sum()), int((near & ~df.correct).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE)
    saved = json.loads(METRICS.read_text(encoding="utf-8"))
    required = ("sample_index", "true_score", "predicted_score", "true_class", "predicted_class")
    if df[list(required)].isna().any().any() or len(df) != 728 or not df.sample_index.is_unique:
        raise AssertionError("Missing or duplicate validation records")
    lower = float(saved["threshold_negative_neutral"])
    upper = float(saved["threshold_neutral_positive"])
    expected_true = np.where(df.true_score < 0, 0, np.where(df.true_score > 0, 2, 1))
    expected_pred = np.where(df.predicted_score < lower, 0,
                             np.where(df.predicted_score > upper, 2, 1))
    if not (np.array_equal(df.true_class, expected_true) and np.array_equal(df.predicted_class, expected_pred)):
        raise AssertionError("Class values disagree with labels or frozen thresholds")
    df["error"] = df.predicted_score - df.true_score
    df["abs_error"] = df.error.abs()
    df["correct"] = df.true_class == df.predicted_class
    cm = pd.crosstab(df.true_class, df.predicted_class).reindex(index=range(3), columns=range(3), fill_value=0).to_numpy()
    f1 = 2 * np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1))
    accuracy = np.trace(cm) / len(df)
    mae = df.abs_error.mean()
    corr = df[["true_score", "predicted_score"]].corr().iloc[0, 1]
    if not (np.array_equal(cm, saved["confusion"]) and
            np.isclose(accuracy, saved["accuracy"]) and
            np.isclose(f1.mean(), saved["macro_f1"]) and
            np.isclose(mae, saved["mae"]) and
            np.isclose(corr, saved["correlation"])):
        raise AssertionError("Saved aggregate metrics do not match per-sample predictions")
    with FEATURES.open("rb") as stream:
        meta = pickle.load(stream)["valid"]
    if len(meta["id"]) != len(df):
        raise AssertionError("Validation metadata length mismatch")
    for row in df.itertuples():
        if not np.isclose(float(meta["regression_labels"][int(row.sample_index)]), row.true_score):
            raise AssertionError(f"Validation row {row.sample_index} differs from source label")

    confusion_plot(cm)
    scatter_plot(df, lower, upper)
    case_indices = [170, 554, 51, 8, 87, 708, 119, 320, 20]
    case_rows = []
    for index in case_indices:
        row = df.loc[df.sample_index == index].iloc[0]
        case_rows.append({"sample_index": index, "sample_id": str(meta["id"][index]),
                          "true_score": float(row.true_score), "predicted_score": float(row.predicted_score),
                          "true_class": int(row.true_class), "predicted_class": int(row.predicted_class),
                          "abs_error": float(row.abs_error), "main_modality": row.main_modality,
                          "main_uncertain": bool(row.main_uncertain),
                          "raw_text": str(meta["raw_text"][index]).replace("\n", " ")})
    with (OUT / "reviewed_error_cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_rows[0].keys())
        writer.writeheader()
        writer.writerows(case_rows)

    true_counts = cm.sum(axis=1)
    miss = len(df) - int(np.trace(cm))
    flips = int(cm[0, 2] + cm[2, 0])
    near_n, near_wrong = near_threshold_fraction(df, lower, upper)
    class_mae = df.groupby("true_class").abs_error.mean()
    class_bias = df.groupby("true_class").error.mean()
    strong = df[np.abs(df.true_score) >= 2]
    lines = [
        "# 问题三验证集性能与错误归因", "",
        "数据：附件2 `aligned_50.pkl` 的 `valid` 划分，共728条；预测来自冻结模型的 `valid_predictions.csv`。",
        "本报告只复核既有输出，不重新训练、选阈值或接触附件4标签。类别0/1/2分别为负向/中性/正向。", "",
        "## 1 基础性能", "",
        "| 指标 | 数值 |", "|---|---:|",
        f"| Accuracy | {accuracy:.4f} |", f"| Macro-F1 | {f1.mean():.4f} |",
        f"| MAE | {mae:.4f} |", f"| Pearson r | {corr:.4f} |", "",
        f"类别阈值：负/中 `{lower:.6f}`；中/正 `{upper:.6f}`。阈值与分类指标均来自同一验证集，故分类成绩是验证集内估计，不是独立测试成绩。", "",
        "| 真实\\预测 | 负向 | 中性 | 正向 | 该类F1 |", "|---|---:|---:|---:|---:|",
    ]
    for i, name in enumerate(("负向", "中性", "正向")):
        lines.append(f"| {name} | {cm[i,0]} | {cm[i,1]} | {cm[i,2]} | {f1[i]:.4f} |")
    lines += ["", "![验证集混淆矩阵](validation_confusion.png)", "", "![真实与预测情感强度](validation_strength_scatter.png)", "",
              "## 2 可观察到的错误形态", "",
              f"- 共 `{miss}` 条极性误判（{miss/len(df):.1%}）；负正两极直接判反 `{flips}` 条，占全部误判的 {flips/miss:.1%}。",
              f"- 中性类 F1 为 `{f1[1]:.4f}`，低于负向 `{f1[0]:.4f}` 和正向 `{f1[2]:.4f}`。真实中性 `{true_counts[1]}` 条中，`{cm[1,0]}` 条被判负向，`{cm[1,2]}` 条被判正向。",
              f"- 按真实类别分组的 MAE：负向 `{class_mae[0]:.4f}`、中性 `{class_mae[1]:.4f}`、正向 `{class_mae[2]:.4f}`；平均预测误差（预测减真实）：负向 `{class_bias[0]:+.4f}`、中性 `{class_bias[1]:+.4f}`、正向 `{class_bias[2]:+.4f}`。两极的平均预测向零靠拢。",
              f"- 真实强度标准差 `{df.true_score.std():.4f}`，预测强度标准差 `{df.predicted_score.std():.4f}`；真实强度绝对值≥2的 `{len(strong)}` 条样本，MAE 为 `{strong.abs_error.mean():.4f}`。这进一步说明强情感的幅度估计偏保守。",
              f"- 以任一分类阈值±0.1为邻域，`{near_n}` 条预测落入邻域，其中 `{near_wrong}` 条误判。阈值附近的类别对小幅分数变化敏感，但它不能解释全部误判。", "",
              "## 3 原文可核对的案例与可能原因", "",
              "以下是诊断性案例，不据此声称已确定模型内部的因果错误机制。完整原文与索引见 `reviewed_error_cases.csv`。", "",
              "| 验证索引 | 真实→预测强度 | 真实→预测类别 | 原文线索 | 可能的错误来源 |", "|---:|---:|---|---|---|",
              "| 170 | -2.000→1.742 | 负→正 | `toilet humor and cheap laughs`；结尾 `highly recommend` | 局部褒义词与整体讽刺/语境相冲突，模型可能过重视后段正向表达。 |",
              "| 51 | 2.333→-0.699 | 正→负 | `out of five stars, I would give it a five` | 正向评分语义未转成相应极性。 |",
              "| 8 | -1.667→0.758 | 负→正 | `except Castaway is really good` | 比较句中被称赞的对象并非当前评价对象。 |",
              "| 87 | 0→-1.381 | 中→负 | `not rated` | 否定词可被误读为负面评价；原标注是中性。 |",
              "| 708 | 0→1.693 | 中→正 | `jean friendly`、`guarantee getting in some fitness time` | 带积极措辞的说明句被判正向，原标注是中性。 |",
              "| 119 | -2.667→-0.011 | 负→中 | `looking at cue cards ... that's how the delivery is` | 负评较依赖完整语境，且该样本主模态判定被标记为不确定。 |",
              "| 20 | 3.000→0.748 | 正→正 | `absolutely awesome` | 类别正确而强度严重低估，说明分类正确不代表回归强度准确。 |", "",
              "这些案例只给出与原文一致的合理假设。要把错误归因提升为更强证据，需要针对相同样本做受控片段遮挡和音视频回看；不能仅凭 `main_modality` 或 Shapley 值断言真实原因。", "",
              "## 4 可直接用于论文的结果表述", "",
              f"在附件2对齐版验证集的728条样本上，模型的情感极性准确率为{accuracy:.4f}，宏平均F1为{f1.mean():.4f}；情感强度MAE为{mae:.4f}，Pearson相关系数为{corr:.4f}。混淆矩阵显示，中性类F1为{f1[1]:.4f}，是三类中最低的。真实中性样本中有{cm[1,0]}条被判负向、{cm[1,2]}条被判正向。分组误差和强度散点图显示，强烈负向与正向的预测幅度均有向零收缩的趋势。典型误判涉及讽刺、比较句的评价对象、否定词和评分语义等情形；这些是基于原文的诊断假设，尚不能据此推断模型内部的因果机制。分类阈值由同一验证集选定，以上分类指标应解释为验证集内性能。", "",
              "## 5 复核说明", "",
              "脚本逐行核对标签符号、冻结阈值、样本唯一性及原始验证集标签，重新计算四项指标与混淆矩阵，并要求其与 `valid_thresholds.json` 一致。图中红点表示极性误判，水平浅橙线为两条冻结分类阈值。", ""]
    (OUT / "validation_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n": len(df), "accuracy": accuracy, "macro_f1": float(f1.mean()),
                      "mae": float(mae), "pearson": float(corr), "confusion": cm.tolist(),
                      "error_count": miss, "extreme_flips": flips,
                      "strong_n": len(strong), "strong_mae": float(strong.abs_error.mean()),
                      "near_threshold_n": near_n, "near_threshold_wrong": near_wrong,
                      "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

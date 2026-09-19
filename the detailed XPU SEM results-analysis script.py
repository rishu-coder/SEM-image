"""Detailed post-processing for the merged Intel-XPU SEM hybrid benchmark.

The script scans completed model folders produced by
``merged_xpu0_sem_hybrid_benchmark.py``, independently validates and recomputes
classification, continuous-stress, and derived-property metrics, then creates
journal-ready 600 dpi PNG and vector PDF figures, CSV tables, a Markdown report,
and an LLM-friendly JSON summary.

This analysis is CPU-side. Training/inference may use Intel XPU GPU 0, but metric
calculation and plotting do not need to reserve XPU memory.

Example
-------
python detailed_results_analysis_xpu0_sem_hybrid.py \
  --root "C:\\Users\\rt4\\Documents\\ML\\CNN\\code\\Combined_XPU_CNN_ViT_Results" \
  --bootstrap 2000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DEFAULT_ROOT = Path(r"C:\Users\rt4\Documents\ML\CNN\code\Combined_XPU_CNN_ViT_Results")
MODEL_LABELS = {
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b_16": "ViT-B/16",
}
STRESS_LEVELS = np.array([100, 200, 400, 800, 1600, 3200, 6000], dtype=float)
PROPERTY_COLUMNS = ["void_ratio", "permeability_m_s", "cv_m2_s", "mv_m2_kN"]
PROPERTY_LABELS = {
    "void_ratio": "Void ratio, e",
    "permeability_m_s": "Permeability, k (m/s)",
    "cv_m2_s": "Coefficient of consolidation, cᵥ (m²/s)",
    "mv_m2_kN": "Volume compressibility, mᵥ (m²/kN)",
}
LOG_PROPERTIES = {"permeability_m_s", "cv_m2_s", "mv_m2_kN"}
COLOURS = sns.color_palette("colorblind", 10)


def set_style():
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    })


def save_figure(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def safe_r2(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return np.nan if len(y) < 2 or np.unique(y).size < 2 else r2_score(y, pred)


def safe_divide(a, b):
    return np.nan if b == 0 else a / b


def model_dirs(root):
    found = [p for p in sorted(root.iterdir())
             if p.is_dir() and (p / "test_predictions.csv").is_file()]
    if not found:
        raise FileNotFoundError(
            f"No model folders containing test_predictions.csv found under {root}"
        )
    return found


def probability_map(df):
    """Return {stress_level: probability column}, accepting benchmark naming."""
    mapping = {}
    for col in df.columns:
        if not col.startswith("probability_"):
            continue
        token = col.removeprefix("probability_").removesuffix("_kpa")
        try:
            mapping[float(token)] = col
        except ValueError:
            pass
    return dict(sorted(mapping.items()))


def read_predictions(model_dir):
    df = pd.read_csv(model_dir / "test_predictions.csv")
    required = [
        "true_stress_kpa", "pred_class_stress_kpa",
        "pred_stress_kpa", "confidence",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{model_dir.name}: missing columns {missing}")
    numeric = required + [c for c in df if c.startswith(("true_", "pred_", "probability_"))
                          and c != "image_path"]
    for col in dict.fromkeys(numeric):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[required].isna().any().any():
        raise ValueError(f"{model_dir.name}: non-numeric or missing core predictions")
    if not np.isfinite(df[required].to_numpy(float)).all():
        raise ValueError(f"{model_dir.name}: non-finite core predictions")
    if ((df.confidence < 0) | (df.confidence > 1)).any():
        raise ValueError(f"{model_dir.name}: confidence outside [0, 1]")
    pmap = probability_map(df)
    if pmap:
        prob = df[list(pmap.values())].to_numpy(float)
        sums = prob.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=2e-3):
            warnings.warn(f"{model_dir.name}: probability rows do not sum to 1 within tolerance")
    return df


def regression_row(model, family, target, y, pred, log_domain=False):
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = np.asarray(y, float)[valid], np.asarray(pred, float)[valid]
    if len(y) == 0:
        return None
    error = pred - y
    row = {
        "model": model, "label": MODEL_LABELS.get(model, model),
        "family": family, "target": target, "n": len(y),
        "r2": safe_r2(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "mae": mean_absolute_error(y, pred),
        "bias": float(error.mean()),
        "median_absolute_error": float(np.median(np.abs(error))),
        "relative_rmse_percent": 100 * safe_divide(mean_squared_error(y, pred) ** 0.5, np.mean(np.abs(y))),
    }
    positive = (y > 0) & (pred > 0)
    if positive.any():
        ly, lp = np.log10(y[positive]), np.log10(pred[positive])
        log_rmse = mean_squared_error(ly, lp) ** 0.5
        row.update({
            "log10_r2": safe_r2(ly, lp),
            "log10_rmse": log_rmse,
            "multiplicative_factor": float(10 ** log_rmse),
        })
    else:
        row.update({"log10_r2": np.nan, "log10_rmse": np.nan,
                    "multiplicative_factor": np.nan})
    row["recommended_domain"] = "log10" if log_domain else "physical"
    return row


def calibration_metrics(df):
    pmap = probability_map(df)
    if not pmap:
        return {"multiclass_log_loss": np.nan, "brier_score": np.nan}
    levels = np.array(list(pmap), dtype=float)
    probs = np.clip(df[list(pmap.values())].to_numpy(float), 1e-12, 1)
    probs /= probs.sum(axis=1, keepdims=True)
    lookup = {level: i for i, level in enumerate(levels)}
    try:
        ids = np.array([lookup[float(x)] for x in df.true_stress_kpa], dtype=int)
    except KeyError:
        return {"multiclass_log_loss": np.nan, "brier_score": np.nan}
    onehot = np.eye(len(levels))[ids]
    return {
        "multiclass_log_loss": log_loss(ids, probs, labels=np.arange(len(levels))),
        "brier_score": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
    }


def expected_calibration_error(df, bins=10):
    correct = df.true_stress_kpa.eq(df.pred_class_stress_kpa).to_numpy(float)
    conf = df.confidence.to_numpy(float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        include = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        count = int(include.sum())
        if count:
            accuracy, confidence = correct[include].mean(), conf[include].mean()
            ece += count / len(df) * abs(accuracy - confidence)
        else:
            accuracy = confidence = np.nan
        rows.append({"bin": i + 1, "lower": edges[i], "upper": edges[i + 1],
                     "count": count, "accuracy": accuracy, "confidence": confidence})
    return float(ece), pd.DataFrame(rows)


def recompute_metrics(model, df):
    true = df.true_stress_kpa.to_numpy(float)
    pred_class = df.pred_class_stress_kpa.to_numpy(float)
    correct = true == pred_class
    ece, reliability = expected_calibration_error(df)
    classification = {
        "model": model, "label": MODEL_LABELS.get(model, model), "n_test": len(df),
        "accuracy": accuracy_score(true, pred_class),
        "balanced_accuracy": balanced_accuracy_score(true, pred_class),
        "macro_f1": f1_score(true, pred_class, average="macro", zero_division=0),
        "weighted_f1": f1_score(true, pred_class, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(true, pred_class),
        "mean_confidence": float(df.confidence.mean()),
        "mean_confidence_correct": float(df.loc[correct, "confidence"].mean()),
        "mean_confidence_incorrect": float(df.loc[~correct, "confidence"].mean()),
        "high_confidence_error_rate": float(((~correct) & (df.confidence >= 0.8)).mean()),
        "ece_10_bins": ece,
        **calibration_metrics(df),
    }
    regression = [regression_row(model, "stress", "stress_kpa", true,
                                 df.pred_stress_kpa.to_numpy(float), True)]
    properties = []
    for target in PROPERTY_COLUMNS:
        true_col, pred_col = f"true_{target}", f"pred_{target}"
        if true_col in df and pred_col in df:
            row = regression_row(model, "property", target,
                                 df[true_col].to_numpy(float),
                                 df[pred_col].to_numpy(float),
                                 target in LOG_PROPERTIES)
            if row:
                properties.append(row)
    reliability.insert(0, "model", model)
    reliability.insert(1, "label", classification["label"])
    return classification, regression, properties, reliability


def bootstrap_metrics(df, iterations, seed):
    rng = np.random.default_rng(seed)
    n = len(df)
    truth = df.true_stress_kpa.to_numpy(float)
    pred_class = df.pred_class_stress_kpa.to_numpy(float)
    pred_stress = df.pred_stress_kpa.to_numpy(float)
    values = np.empty((iterations, 6), dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        y, pc, ps = truth[idx], pred_class[idx], pred_stress[idx]
        ly, lp = np.log10(y), np.log10(np.clip(ps, np.finfo(float).tiny, None))
        values[i] = [
            accuracy_score(y, pc),
            balanced_accuracy_score(y, pc),
            f1_score(y, pc, average="macro", zero_division=0),
            safe_r2(ly, lp),
            mean_squared_error(ly, lp) ** 0.5,
            mean_absolute_error(y, ps),
        ]
    result = {}
    names = ["accuracy", "balanced_accuracy", "macro_f1", "stress_log10_r2",
             "stress_log10_rmse", "stress_mae"]
    for j, name in enumerate(names):
        finite = values[:, j][np.isfinite(values[:, j])]
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = (
            np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        )
    return result


def class_specific_metrics(predictions):
    rows = []
    for model, df in predictions.items():
        classes = sorted(set(df.true_stress_kpa) | set(df.pred_class_stress_kpa))
        for level in classes:
            y = df.true_stress_kpa.eq(level)
            pred = df.pred_class_stress_kpa.eq(level)
            tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
            rows.append({
                "model": model, "label": MODEL_LABELS.get(model, model),
                "stress_kpa": level, "support": int(y.sum()),
                "precision": safe_divide(tp, tp + fp), "recall": safe_divide(tp, tp + fn),
                "f1": f1_score(y, pred, zero_division=0),
            })
    return pd.DataFrame(rows)


def per_level_regression(predictions):
    rows = []
    for model, df in predictions.items():
        for level, group in df.groupby("true_stress_kpa"):
            y, pred = group.true_stress_kpa.to_numpy(float), group.pred_stress_kpa.to_numpy(float)
            log_error = np.log10(np.clip(pred, 1e-12, None)) - np.log10(y)
            rows.append({
                "model": model, "label": MODEL_LABELS.get(model, model),
                "stress_kpa": level, "n": len(group),
                "mean_prediction_kpa": pred.mean(), "median_prediction_kpa": np.median(pred),
                "mae_kpa": mean_absolute_error(y, pred), "bias_kpa": float((pred - y).mean()),
                "log10_rmse": float(np.sqrt(np.mean(log_error ** 2))),
                "multiplicative_factor": float(10 ** np.sqrt(np.mean(log_error ** 2))),
            })
    return pd.DataFrame(rows)


def training_summary(model, model_dir):
    path = model_dir / "training_history.csv"
    if not path.is_file():
        return None
    h = pd.read_csv(path)
    if h.empty or "val_loss" not in h:
        return None
    best_idx = h.val_loss.idxmin()
    seconds_col = "seconds" if "seconds" in h else None
    total_seconds = float(h[seconds_col].sum()) if seconds_col else np.nan
    return {
        "model": model, "label": MODEL_LABELS.get(model, model),
        "epochs_completed": len(h),
        "best_global_epoch": int(h.loc[best_idx, "global_epoch"]) if "global_epoch" in h else int(best_idx + 1),
        "best_val_loss": float(h.loc[best_idx, "val_loss"]),
        "final_train_loss": float(h.iloc[-1].train_loss),
        "final_val_loss": float(h.iloc[-1].val_loss),
        "final_generalisation_gap": float(h.iloc[-1].val_loss - h.iloc[-1].train_loss),
        "training_seconds": total_seconds,
        "training_minutes": total_seconds / 60 if np.isfinite(total_seconds) else np.nan,
    }


def plot_overview(class_df, reg_df, property_df, train_df, out):
    order = class_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).model.tolist()
    labels = dict(zip(class_df.model, class_df.label))
    palette = dict(zip(order, COLOURS[:len(order)]))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    long = class_df.melt(id_vars=["model", "label"],
                         value_vars=["accuracy", "balanced_accuracy", "macro_f1"],
                         var_name="metric", value_name="score")
    sns.barplot(data=long, x="metric", y="score", hue="model", hue_order=order,
                palette=palette, ax=axes[0])
    axes[0].set(xlabel="", ylabel="Score", ylim=(0, 1), title="(a) Classification")
    axes[0].set_xticklabels(["Accuracy", "Balanced\naccuracy", "Macro F1"])
    stress = reg_df[reg_df.target == "stress_kpa"].set_index("model").reindex(order)
    axes[1].barh([labels[m] for m in stress.index], stress.log10_r2,
                 color=[palette[m] for m in stress.index])
    axes[1].invert_yaxis(); axes[1].set(xlabel="log₁₀-stress R²", title="(b) Continuous stress")
    prop = property_df.groupby("model").r2.mean().reindex(order)
    axes[2].barh([labels[m] for m in prop.index], prop,
                 color=[palette[m] for m in prop.index])
    axes[2].invert_yaxis(); axes[2].set(xlabel="Mean physical-domain R²", title="(c) Derived properties")
    for ax in axes:
        ax.grid(False); sns.despine(ax=ax)
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, [labels[m] for m in order], frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.58), ncol=2)
    fig.subplots_adjust(bottom=0.31, wspace=0.48)
    save_figure(fig, out, "figure_01_model_performance_overview")


def plot_confusions(predictions, out):
    classes = sorted(set().union(*[set(df.true_stress_kpa) for df in predictions.values()]))
    n, cols = len(predictions), 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.3 * rows), squeeze=False)
    for ax, (model, df) in zip(axes.flat, predictions.items()):
        cm = confusion_matrix(df.true_stress_kpa, df.pred_class_stress_kpa,
                              labels=classes, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    cbar=False, xticklabels=[f"{x:g}" for x in classes],
                    yticklabels=[f"{x:g}" for x in classes], ax=ax)
        ax.set(xlabel="Predicted stress class (kPa)", ylabel="Measured stress class (kPa)",
               title=MODEL_LABELS.get(model, model))
    for ax in axes.flat[n:]: ax.axis("off")
    fig.tight_layout(); save_figure(fig, out, "figure_02_normalised_confusion_matrices")


def plot_stress_parity(predictions, out):
    n, cols = len(predictions), 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9.5, 4 * rows), squeeze=False)
    lo, hi = np.inf, -np.inf
    for df in predictions.values():
        lo = min(lo, df.true_stress_kpa.min(), df.pred_stress_kpa.min())
        hi = max(hi, df.true_stress_kpa.max(), df.pred_stress_kpa.max())
    for ax, (model, df) in zip(axes.flat, predictions.items()):
        ax.scatter(df.true_stress_kpa, df.pred_stress_kpa, s=18, alpha=.55,
                   color="#0072B2", edgecolors="white", linewidth=.3)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ly, lp = np.log10(df.true_stress_kpa), np.log10(np.clip(df.pred_stress_kpa, 1e-12, None))
        rmse = mean_squared_error(ly, lp) ** .5
        ax.text(.04, .96, f"log₁₀ R² = {safe_r2(ly, lp):.3f}\nFactor = {10**rmse:.2f}×",
                transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", fc="white", alpha=.9))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set(xlabel="Measured stress (kPa)", ylabel="Predicted stress (kPa)",
               title=MODEL_LABELS.get(model, model))
        ax.grid(False); sns.despine(ax=ax)
    for ax in axes.flat[n:]: ax.axis("off")
    fig.tight_layout(); save_figure(fig, out, "figure_03_continuous_stress_parity")


def plot_property_heatmaps(property_df, out):
    for metric, title, filename in [
        ("r2", "Derived-property R²", "figure_04_property_r2_heatmap"),
        ("relative_rmse_percent", "Derived-property relative RMSE (%)", "figure_05_property_relative_rmse_heatmap"),
    ]:
        pivot = property_df.pivot(index="label", columns="target", values=metric)
        pivot = pivot.rename(columns=PROPERTY_LABELS)
        fig, ax = plt.subplots(figsize=(8.2, max(3, .48 * len(pivot) + 1.5)))
        cmap = "viridis" if metric == "r2" else "YlOrRd"
        sns.heatmap(pivot, annot=True, fmt=".3f" if metric == "r2" else ".1f",
                    cmap=cmap, linewidths=.4, cbar_kws={"label": title}, ax=ax)
        ax.set(xlabel="Target", ylabel="Model", title=title)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); save_figure(fig, out, filename)


def plot_per_level(per_level_df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    sns.lineplot(data=per_level_df, x="stress_kpa", y="mae_kpa", hue="label",
                 marker="o", palette="colorblind", ax=axes[0])
    sns.lineplot(data=per_level_df, x="stress_kpa", y="multiplicative_factor", hue="label",
                 marker="s", palette="colorblind", ax=axes[1], legend=False)
    for ax in axes:
        ax.set_xscale("log"); ax.grid(False); sns.despine(ax=ax)
    axes[0].set(xlabel="Measured stress (kPa)", ylabel="MAE (kPa)", title="(a) Physical-domain error")
    axes[1].set(xlabel="Measured stress (kPa)", ylabel="Multiplicative error factor", title="(b) Log-domain error")
    axes[0].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.subplots_adjust(wspace=.55); save_figure(fig, out, "figure_06_stress_error_by_level")


def plot_learning_curves(dirs, out):
    valid = []
    for d in dirs:
        path = d / "training_history.csv"
        if path.is_file():
            h = pd.read_csv(path)
            if not h.empty:
                valid.append((d.name, h))
    if not valid: return
    colours = dict(zip([m for m, _ in valid], sns.color_palette("colorblind", len(valid))))
    panels = [
        ("loss", "Total loss", "Loss"),
        ("accuracy", "Classification accuracy", "Accuracy"),
        ("classification_loss", "Classification loss", "Loss"),
        ("regression_loss", "Stress-regression loss", "Huber loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        plotted = False
        for model, h in valid:
            x = h.global_epoch if "global_epoch" in h else np.arange(1, len(h) + 1)
            for prefix, ls in [("train", "-"), ("val", "--")]:
                col = f"{prefix}_{metric}"
                if col in h:
                    ax.plot(x, h[col], color=colours[model], ls=ls, lw=1.4,
                            label=MODEL_LABELS.get(model, model) if prefix == "train" else None)
                    plotted = True
            if "stage" in h:
                transitions = h.index[h.stage.ne(h.stage.shift())].tolist()[1:]
                for idx in transitions:
                    ax.axvline(x.iloc[idx] - .5, color=colours[model], ls=":", lw=.6, alpha=.4)
        ax.set(xlabel="Global epoch", ylabel=ylabel, title=title)
        if metric == "accuracy": ax.set_ylim(0, 1)
        if not plotted: ax.text(.5, .5, "Metric not recorded", ha="center", va="center", transform=ax.transAxes)
        ax.grid(False); sns.despine(ax=ax)
    model_handles = [mpl.lines.Line2D([0], [0], color=colours[m], lw=2,
                     label=MODEL_LABELS.get(m, m)) for m, _ in valid]
    style_handles = [mpl.lines.Line2D([0], [0], color="0.2", ls="-", label="Training"),
                     mpl.lines.Line2D([0], [0], color="0.2", ls="--", label="Validation"),
                     mpl.lines.Line2D([0], [0], color="0.4", ls=":", label="Fine-tuning starts")]
    fig.legend(handles=model_handles + style_handles, loc="lower center",
               bbox_to_anchor=(.5, -.01), ncol=5, frameon=False)
    fig.subplots_adjust(bottom=.18, hspace=.35, wspace=.28)
    save_figure(fig, out, "figure_07_merged_training_curves")


def plot_reliability(reliability_df, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for label, g in reliability_df.groupby("label"):
        g = g[g["count"] > 0]
        ax.plot(g.confidence, g.accuracy, marker="o", lw=1.3, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1),
           title="Reliability diagram")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(False); sns.despine(ax=ax); fig.tight_layout()
    save_figure(fig, out, "figure_08_reliability_diagram")


def plot_accuracy_cost(class_df, train_df, out):
    if train_df.empty or train_df.training_minutes.isna().all(): return
    data = class_df.merge(train_df[["model", "training_minutes"]], on="model", how="inner")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(data.training_minutes, data.macro_f1, s=75, color=COLOURS[:len(data)])
    for row in data.itertuples():
        ax.annotate(row.label, (row.training_minutes, row.macro_f1),
                    xytext=(5, 4), textcoords="offset points")
    ax.set(xlabel="Training time (min)", ylabel="Macro F1", title="Performance and computational cost")
    ax.grid(False); sns.despine(ax=ax); fig.tight_layout()
    save_figure(fig, out, "figure_09_performance_cost")


def target_linkage_diagnostics(predictions):
    """Check whether properties are deterministic functions of true stress."""
    if not predictions: return pd.DataFrame()
    _, df = next(iter(predictions.items()))
    rows = []
    for target in PROPERTY_COLUMNS:
        col = f"true_{target}"
        if col not in df: continue
        grouped = df.groupby("true_stress_kpa")[col]
        unique = grouped.nunique(dropna=False)
        std = grouped.std().fillna(0)
        rows.append({
            "target": target,
            "maximum_unique_values_within_stress_level": int(unique.max()),
            "maximum_within_level_std": float(std.max()),
            "fixed_within_every_stress_level": bool((unique <= 1).all()),
        })
    return pd.DataFrame(rows)


def fmt(value, digits=4):
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def write_report(class_df, regression_df, property_df, train_df, class_detail,
                 per_level_df, linkage_df, out):
    ranked = class_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).reset_index(drop=True)
    stress = regression_df[regression_df.target == "stress_kpa"].copy()
    property_mean = property_df.groupby(["model", "label"], as_index=False).agg(
        mean_r2=("r2", "mean"), mean_relative_rmse_percent=("relative_rmse_percent", "mean"),
        mean_mae=("mae", "mean")) if not property_df.empty else pd.DataFrame()
    best_stress = stress.sort_values("log10_r2", ascending=False).iloc[0]
    best_property = property_mean.sort_values("mean_r2", ascending=False).iloc[0] if not property_mean.empty else None
    fixed = bool(len(linkage_df) and linkage_df.fixed_within_every_stress_level.all())

    lines = [
        "# Comprehensive Analysis of the Intel-XPU SEM Hybrid Benchmark", "",
        "## 1. Scope", "",
        f"- Models analysed: **{len(class_df)}**.",
        f"- Test observations per model: **{int(class_df.n_test.min())} to {int(class_df.n_test.max())}**.",
        "- Tasks: discrete consolidation-stress classification, continuous stress regression, and four stress-derived engineering-property estimates.",
        "- Uncertainty: paired non-parametric 95% bootstrap intervals.",
        "- Graphics: 600 dpi PNG and vector PDF.", "",
        "## 2. Executive summary", "",
        f"- Best classification: **{ranked.iloc[0].label}**, macro F1 {fmt(ranked.iloc[0].macro_f1)}, balanced accuracy {fmt(ranked.iloc[0].balanced_accuracy)}, accuracy {fmt(ranked.iloc[0].accuracy)}, MCC {fmt(ranked.iloc[0].mcc)}.",
        f"- Best continuous stress model by log-domain R²: **{best_stress.label}**, log₁₀ R² {fmt(best_stress.log10_r2)}, multiplicative error factor {fmt(best_stress.multiplicative_factor, 2)}×.",
    ]
    if best_property is not None:
        lines.append(f"- Best mean derived-property R²: **{best_property.label}**, {fmt(best_property.mean_r2)}.")
    if len(train_df) and train_df.training_minutes.notna().any():
        lines.append(f"- Recorded training time across models: **{train_df.training_minutes.sum():.1f} min**.")

    lines += ["", "## 3. Classification comparison", "",
              "|Rank|Model|Accuracy|Balanced accuracy|Macro F1|MCC|Macro F1 95% CI|ECE|Log loss|",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, row in ranked.iterrows():
        lines.append(f"|{i+1}|{row.label}|{fmt(row.accuracy)}|{fmt(row.balanced_accuracy)}|{fmt(row.macro_f1)}|{fmt(row.mcc)}|{fmt(row.macro_f1_ci_low)} to {fmt(row.macro_f1_ci_high)}|{fmt(row.ece_10_bins)}|{fmt(row.multiclass_log_loss)}|")

    lines += ["", "## 4. Continuous-stress regression", "",
              "|Model|Physical R²|Physical RMSE (kPa)|MAE (kPa)|Bias (kPa)|log₁₀ R²|Multiplicative factor|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in stress.sort_values("log10_r2", ascending=False).itertuples():
        lines.append(f"|{row.label}|{fmt(row.r2)}|{fmt(row.rmse, 2)}|{fmt(row.mae, 2)}|{fmt(row.bias, 2)}|{fmt(row.log10_r2)}|{fmt(row.multiplicative_factor, 2)}×|")

    lines += ["", "## 5. Derived engineering properties", ""]
    if property_mean.empty:
        lines.append("No complete derived-property prediction columns were found.")
    else:
        lines += ["|Model|Mean R²|Mean relative RMSE (%)|Mean raw MAE*|",
                  "|---|---:|---:|---:|"]
        for row in property_mean.sort_values("mean_r2", ascending=False).itertuples():
            lines.append(f"|{row.label}|{fmt(row.mean_r2)}|{fmt(row.mean_relative_rmse_percent, 2)}|{fmt(row.mean_mae)}|")
        lines += ["", "*Mean raw MAE combines differently scaled physical quantities and is descriptive only.", "", "Target-specific leaders:"]
        for target in PROPERTY_COLUMNS:
            subset = property_df[property_df.target == target].sort_values("r2", ascending=False)
            if len(subset):
                row = subset.iloc[0]
                lines.append(f"- **{PROPERTY_LABELS[target]}**: {row.label}, R²={fmt(row.r2)}, RMSE={fmt(row.rmse)}, MAE={fmt(row.mae)}, bias={fmt(row.bias)}.")

    lines += ["", "## 6. Class-level diagnostics", ""]
    for model in ranked.model:
        subset = class_detail[class_detail.model == model].sort_values("f1")
        if len(subset):
            weak, strong = subset.iloc[0], subset.iloc[-1]
            lines.append(f"- **{MODEL_LABELS.get(model, model)}**: strongest stress level {strong.stress_kpa:g} kPa (F1={fmt(strong.f1)}); weakest {weak.stress_kpa:g} kPa (F1={fmt(weak.f1)}).")

    lines += ["", "## 7. Training and generalisation", "",
              "|Model|Epochs|Best epoch|Best validation loss|Final train loss|Final validation loss|Final gap|Training time (min)|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    if train_df.empty:
        lines.append("|No training-history files found|NA|NA|NA|NA|NA|NA|NA|")
    else:
        for row in train_df.sort_values("best_val_loss").itertuples():
            lines.append(f"|{row.label}|{row.epochs_completed}|{row.best_global_epoch}|{fmt(row.best_val_loss)}|{fmt(row.final_train_loss)}|{fmt(row.final_val_loss)}|{fmt(row.final_generalisation_gap)}|{fmt(row.training_minutes, 1)}|")

    lines += ["", "## 8. Scientific interpretation", "",
              f"- All engineering-property targets fixed within each measured stress level: **{fixed}**."]
    if fixed:
        lines += [
            "- The property outputs are deterministic stress-conditioned values produced from the supplied interpolation table.",
            "- Their apparent predictive performance is therefore inherited from continuous-stress prediction and should not be presented as independent specimen-level property measurement.",
            "- Independent property prediction would require directly measured property targets with genuine within-stress variation.",
        ]
    lines += [
        "- Continuous stress spans orders of magnitude, so log-domain R², log RMSE, and multiplicative error factor should accompany physical-domain RMSE and MAE.",
        "- Balanced accuracy and macro F1 should be emphasised if stress-level frequencies are unequal.",
        "- Calibration statistics describe whether confidence values are trustworthy, not only whether predicted classes are correct.",
        "- Conclusions remain limited to the held-out dataset. External batches, instruments, preparation conditions, and magnifications require validation.",
        "", "## 9. Manuscript-ready outputs", "",
        "- `figure_01_model_performance_overview`: compact task-level comparison.",
        "- `figure_02_normalised_confusion_matrices`: stress-class error patterns.",
        "- `figure_03_continuous_stress_parity`: measured versus predicted stress.",
        "- `figure_04` and `figure_05`: target-specific engineering-property heatmaps.",
        "- `figure_06_stress_error_by_level`: heteroscedasticity across stress levels.",
        "- `figure_07_merged_training_curves`: convergence and fine-tuning behaviour.",
        "- `figure_08_reliability_diagram`: probability calibration.",
        "- `figure_09_performance_cost`: macro F1 versus training time.",
        "- `table_01` to `table_09`: reusable machine-readable statistics.",
    ]
    (out / "comprehensive_results_report.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "analysis_scope": {"models": len(class_df), "test_size_min": int(class_df.n_test.min()),
                           "test_size_max": int(class_df.n_test.max())},
        "best_classification": ranked.iloc[0].replace({np.nan: None}).to_dict(),
        "best_continuous_stress": best_stress.replace({np.nan: None}).to_dict(),
        "properties_fixed_within_stress_level": fixed,
        "classification_ranking": ranked.replace({np.nan: None}).to_dict(orient="records"),
        "regression_metrics": regression_df.replace({np.nan: None}).to_dict(orient="records"),
        "property_metrics": property_df.replace({np.nan: None}).to_dict(orient="records"),
        "training_summary": train_df.replace({np.nan: None}).to_dict(orient="records"),
    }
    (out / "comprehensive_results_for_llm.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def write_plain_summary(class_df, regression_df, property_df, train_df, out):
    best = class_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
    stress = regression_df[regression_df.target == "stress_kpa"].sort_values("log10_r2", ascending=False).iloc[0]
    lines = [
        "AUTOMATED RESULTS SUMMARY", "=" * 80,
        f"Models analysed: {len(class_df)}",
        f"Test observations per model: {class_df.n_test.min():.0f} to {class_df.n_test.max():.0f}", "",
        f"Best classification: {best.label} (macro F1={best.macro_f1:.4f}, balanced accuracy={best.balanced_accuracy:.4f}, accuracy={best.accuracy:.4f}).",
        f"Best continuous stress: {stress.label} (log10 R2={stress.log10_r2:.4f}, multiplicative factor={stress.multiplicative_factor:.2f}x).",
        "",
        "Interpretation: engineering properties are interpolated from predicted stress. They are",
        "stress-conditioned outputs rather than independently measured specimen-level predictions.",
    ]
    if len(train_df) and train_df.training_minutes.notna().any():
        lines += ["", f"Total recorded training time: {train_df.training_minutes.sum():.1f} min."]
    (out / "publication_results_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main(root, bootstrap, seed):
    set_style()
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {root}")
    out = root / "Publication_Ready_Detailed_Analysis"
    out.mkdir(parents=True, exist_ok=True)

    dirs = model_dirs(root)
    predictions, class_rows, regression_rows, property_rows = {}, [], [], []
    reliability_rows, training_rows, warnings_list = [], [], []
    for model_dir in dirs:
        try:
            df = read_predictions(model_dir)
            predictions[model_dir.name] = df
            classification, regression, properties, reliability = recompute_metrics(model_dir.name, df)
            classification.update(bootstrap_metrics(df, bootstrap, seed))
            class_rows.append(classification)
            regression_rows.extend(regression)
            property_rows.extend(properties)
            reliability_rows.append(reliability)
            summary = training_summary(model_dir.name, model_dir)
            if summary: training_rows.append(summary)
        except Exception as exc:
            warnings_list.append(f"{model_dir.name}: {type(exc).__name__}: {exc}")

    if not class_rows:
        raise RuntimeError("No complete model results could be analysed. See validation errors.")

    class_df = pd.DataFrame(class_rows).sort_values("macro_f1", ascending=False)
    regression_df = pd.DataFrame(regression_rows)
    property_df = pd.DataFrame(property_rows)
    reliability_df = pd.concat(reliability_rows, ignore_index=True)
    train_df = pd.DataFrame(training_rows)
    class_detail = class_specific_metrics(predictions)
    per_level_df = per_level_regression(predictions)
    linkage_df = target_linkage_diagnostics(predictions)

    class_df.to_csv(out / "table_01_classification_comparison.csv", index=False)
    regression_df.to_csv(out / "table_02_continuous_stress_metrics.csv", index=False)
    property_df.to_csv(out / "table_03_property_metrics_all_models.csv", index=False)
    train_df.to_csv(out / "table_04_training_summary.csv", index=False)
    class_detail.to_csv(out / "table_05_class_specific_metrics.csv", index=False)
    per_level_df.to_csv(out / "table_06_stress_metrics_by_level.csv", index=False)
    reliability_df.to_csv(out / "table_07_reliability_bins.csv", index=False)
    linkage_df.to_csv(out / "table_08_target_linkage_diagnostics.csv", index=False)

    ranking = class_df[["model", "label", "accuracy", "balanced_accuracy", "macro_f1", "mcc",
                        "ece_10_bins", "multiclass_log_loss"]].copy()
    stress_lookup = regression_df[regression_df.target == "stress_kpa"].set_index("model")
    ranking["stress_log10_r2"] = ranking.model.map(stress_lookup.log10_r2)
    ranking["stress_multiplicative_factor"] = ranking.model.map(stress_lookup.multiplicative_factor)
    if not property_df.empty:
        ranking["mean_property_r2"] = ranking.model.map(property_df.groupby("model").r2.mean())
        ranking["mean_property_relative_rmse_percent"] = ranking.model.map(
            property_df.groupby("model").relative_rmse_percent.mean())
    ranking["classification_rank"] = ranking.macro_f1.rank(ascending=False, method="min").astype(int)
    ranking.to_csv(out / "table_09_consolidated_model_ranking.csv", index=False)

    plot_overview(class_df, regression_df, property_df, train_df, out)
    plot_confusions(predictions, out)
    plot_stress_parity(predictions, out)
    if not property_df.empty: plot_property_heatmaps(property_df, out)
    plot_per_level(per_level_df, out)
    plot_learning_curves(dirs, out)
    plot_reliability(reliability_df, out)
    plot_accuracy_cost(class_df, train_df, out)
    write_plain_summary(class_df, regression_df, property_df, train_df, out)
    write_report(class_df, regression_df, property_df, train_df, class_detail,
                 per_level_df, linkage_df, out)
    (out / "analysis_warnings.txt").write_text(
        "\n".join(warnings_list) if warnings_list else "No warnings.\n", encoding="utf-8"
    )

    print(f"Analysed {len(class_df)} models")
    print(f"Detailed publication outputs: {out}")
    print(class_df[["label", "accuracy", "balanced_accuracy", "macro_f1", "mcc",
                    "ece_10_bins"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create detailed journal-ready analysis from the merged Intel-XPU SEM benchmark."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="Root containing one model folder per completed benchmark model")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap repetitions for 95 percent confidence intervals")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.root, args.bootstrap, args.seed)

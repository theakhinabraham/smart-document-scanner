"""Evaluate the scanner on all annotated photos and produce every number and figure for the report.

Usage:
    python evaluate.py                                   # real photos in data/raw
    python evaluate.py --iou 0.85                        # different success threshold
    python evaluate.py --images data/samples --gt data/samples/ground_truth.csv \
        --conditions data/samples/conditions.csv --out results/samples   # synthetic demo

Outputs (in --out, default results/):
    results.csv                 one row per photo: method, IoU, success, conditions, time
    summary.md                  overall / per-condition / per-method tables (Markdown)
    summary_by_condition.csv    the same tables as CSV
    summary_by_method.csv
    failure_analysis.md         every failure with conditions + auto-generated likely cause
    figures/fig1_pipeline_stages.png
    figures/fig2_before_after.png
    figures/fig3_failure_gallery.png
    figures/fig4_success_by_condition.png
    figures/fig5_iou_distribution.png

How success is measured: the detected page outline (a quadrilateral) is
compared with the hand-clicked ground-truth outline using Intersection over
Union (IoU). A photo counts as a success if IoU >= 0.90 (configurable).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # draw to files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scanner import draw_outline, load_image, polygon_iou, scan_document  # noqa: E402
from scanner.metrics import polygon_area  # noqa: E402
from scanner.pipeline import list_images  # noqa: E402

CONDITION_COLUMNS = ["angle", "lighting", "background", "page"]
CONDITION_VALUES = {  # allowed values, also used to order the bar chart
    "angle": ["straight", "tilted", "steep"],
    "lighting": ["good", "dim", "shadow", "glare"],
    "background": ["plain", "cluttered", "low-contrast"],
    "page": ["flat", "curled", "folded"],
}
METHOD_ORDER = ["canny", "otsu", "minarearect", "failed"]

# Figure colours: one accent hue for single-series bars; red/green outlines as in the brief.
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
RED_BGR, GREEN_BGR = (40, 40, 230), (40, 170, 40)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})


# --- Reading inputs -------------------------------------------------------------

def read_ground_truth(path):
    """Read ground_truth.csv into {filename: (4, 2) array of corners TL, TR, BR, BL}."""
    truth = {}
    path = Path(path)
    if not path.exists():
        return truth
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                values = [float(row[f"{axis}{i}"]) for i in range(1, 5) for axis in ("x", "y")]
            except (KeyError, ValueError, TypeError):
                continue  # incomplete row
            truth[row["filename"]] = np.array(values, dtype=np.float32).reshape(4, 2)
    return truth


def read_conditions(path):
    """Read conditions.csv into a DataFrame indexed by filename (blank cells become '')."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CONDITION_COLUMNS)
    df = pd.read_csv(path, dtype=str).fillna("")
    for column in CONDITION_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].str.strip().str.lower()
    return df.set_index("filename")[CONDITION_COLUMNS]


# --- Small image helpers ----------------------------------------------------------

def thumbnail(image, max_side=900):
    """Shrink an image so its longest side is at most max_side. Returns (thumb, scale)."""
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, scale


def to_rgb(image):
    """BGR or grayscale -> RGB for matplotlib."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show(ax, image, title=None):
    ax.imshow(to_rgb(image))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, color=INK)


# --- Failure analysis -------------------------------------------------------------

def likely_cause(method, iou, detected, truth, conditions):
    """Generate a short, plausible explanation for a failure (to be checked by hand).

    The geometric part compares the detected and true quadrilaterals:
      * intersection area is recovered from IoU:  I = IoU * (A + B) / (1 + IoU)
      * if the detection is much SMALLER than the page and lies inside it,
        an inner contour (text block, table, fold) was probably chosen;
      * if it is much LARGER and contains the page, the page outline probably
        merged with a background object;
      * otherwise the quad is roughly the right size but misplaced.
    The condition part adds a hint for each hard condition recorded for the photo.
    """
    causes = []
    if method == "failed":
        causes.append("no page outline found by any method (Canny, Otsu and minAreaRect all failed)")
    elif truth is not None:
        det_area, gt_area = polygon_area(detected), polygon_area(truth)
        inter = iou * (det_area + gt_area) / (1 + iou) if iou > 0 else 0.0
        det_inside_gt = inter / det_area if det_area else 0.0
        gt_inside_det = inter / gt_area if gt_area else 0.0
        size = det_area / gt_area if gt_area else 0.0
        if size < 0.85 and det_inside_gt > 0.9:
            causes.append(f"detected quad covers less area than ground truth ({size:.0%} of the page): "
                          "likely picked an inner contour (text block, table, frame or fold line)")
        elif size > 1.15 and gt_inside_det > 0.9:
            causes.append(f"detected quad is larger than the page ({size:.0%}): page edge merged "
                          "with a background object, shadow or table edge")
        else:
            causes.append("detected quad is roughly page-sized but misplaced: one or more corners "
                          "snapped to a shadow, clutter or glare edge")
        if method == "minarearect":
            causes.append("fell back to minAreaRect: the outline was not a clean 4-sided polygon, "
                          "and a rotated rectangle cannot follow perspective")

    hints = {
        ("background", "low-contrast"): "low-contrast background: page edge too faint for Canny "
                                        "(brightness step below the thresholds)",
        ("background", "cluttered"): "cluttered background: objects touching the page break or "
                                     "extend its outline",
        ("lighting", "shadow"): "shadow: the shadow boundary is a strong edge that can be "
                                "mistaken for the page border",
        ("lighting", "glare"): "glare: over-exposed area washes out part of the page border",
        ("lighting", "dim"): "dim lighting: weaker edges and more sensor noise",
        ("page", "curled"): "curled page: border is curved, so a 4-corner model fits poorly",
        ("page", "folded"): "folded page: the crease creates an extra edge inside the page",
        ("angle", "steep"): "steep angle: far edge is short, blurry and strongly foreshortened",
    }
    for column in CONDITION_COLUMNS:
        hint = hints.get((column, conditions.get(column, "")))
        if hint:
            causes.append(hint)
    return "; ".join(causes) if causes else "unclear - inspect the image"


def write_failure_analysis(df, path, iou_threshold):
    """Write failure_analysis.md listing every failure with its conditions and likely cause."""
    failures = df[df["success"] == False]  # noqa: E712  (NaN = not annotated, excluded)
    lines = [
        "# Failure analysis",
        "",
        f"Success criterion: IoU >= {iou_threshold:.2f} between detected and ground-truth page outline.",
        f"Failures: {len(failures)} of {int(df['success'].notna().sum())} annotated photos.",
        "",
        "The *likely cause* lines were generated automatically from the geometry of the "
        "detection and the recorded conditions. Check each one against the image in "
        "`figures/fig3_failure_gallery.png` (red = detected, green = ground truth) and edit "
        "the text below by hand.",
        "",
    ]
    if failures.empty:
        lines.append("No failures. (Consider adding harder photos so the analysis has substance.)")
    for _, row in failures.iterrows():
        conditions = ", ".join(f"{c}={row[c] or '?'}" for c in CONDITION_COLUMNS)
        lines += [
            f"## {row['filename']}",
            "",
            f"- **Method:** {row['method']}",
            f"- **IoU:** {row['iou']:.3f}",
            f"- **Conditions:** {conditions}",
            f"- **Likely cause (auto):** {row['likely_cause']}",
            "- **Notes (manual):** _TODO: confirm or correct the cause; suggest a fix._",
            "",
        ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# --- Summary tables -----------------------------------------------------------------

def group_table(df, column, order=None):
    """Success rate, count and mean IoU for each value of `column` (annotated photos only)."""
    data = df[df["success"].notna()].copy()
    data["success"] = data["success"].astype(bool)
    data = data[data[column].astype(str) != ""]
    if data.empty:
        return pd.DataFrame(columns=[column, "n", "successes", "success_rate", "mean_iou"])
    table = (data.groupby(column)
             .agg(n=("success", "size"), successes=("success", "sum"), mean_iou=("iou", "mean"))
             .reset_index())
    table["successes"] = table["successes"].astype(int)
    table["success_rate"] = table["successes"] / table["n"]
    if order:
        rank = {v: i for i, v in enumerate(order)}
        table = table.sort_values(column, key=lambda s: s.map(lambda v: rank.get(v, 99)))
    return table[[column, "n", "successes", "success_rate", "mean_iou"]]


def to_markdown(table):
    """Format a summary table as a Markdown table (no extra dependency needed)."""
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = []
    for _, r in table.iterrows():
        cells = []
        for col in table.columns:
            v = r[col]
            if col == "success_rate":
                cells.append(f"{v:.0%}")
            elif col == "mean_iou":
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule] + rows)


def write_summary(df, out_dir, iou_threshold):
    """Print and save the overall / per-condition / per-method summary. Returns the text."""
    annotated = df[df["success"].notna()]
    n = len(annotated)
    successes = int(annotated["success"].astype(bool).sum()) if n else 0
    lines = ["# Evaluation summary", ""]
    lines.append(f"- Photos processed: {len(df)} (annotated with ground truth: {n})")
    if n:
        lines.append(f"- **Overall success rate (IoU >= {iou_threshold:.2f}): "
                     f"{successes}/{n} = {successes / n:.1%}**")
        lines.append(f"- Mean IoU: {annotated['iou'].mean():.3f} "
                     f"(median {annotated['iou'].median():.3f})")
    lines.append(f"- Mean processing time: {df['time_ms'].mean():.0f} ms per photo")
    missing = df[df["success"].isna()]["filename"].tolist()
    if missing:
        lines.append(f"- Not annotated (excluded from rates): {', '.join(missing)}")
    lines.append("")

    method_table = group_table(df, "method", METHOD_ORDER)
    lines += ["## By detection method", "", to_markdown(method_table), ""]
    method_table.to_csv(out_dir / "summary_by_method.csv", index=False)

    condition_tables = []
    lines += ["## By condition", ""]
    for column in CONDITION_COLUMNS:
        table = group_table(df, column, CONDITION_VALUES[column])
        if table.empty:
            lines += [f"_{column}: not filled in conditions.csv_", ""]
            continue
        lines += [f"### {column}", "", to_markdown(table), ""]
        condition_tables.append(table.rename(columns={column: "value"}).assign(condition=column))
    if condition_tables:
        combined = pd.concat(condition_tables)[["condition", "value", "n", "successes",
                                               "success_rate", "mean_iou"]]
        combined.to_csv(out_dir / "summary_by_condition.csv", index=False)

    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return text


# --- Figures ------------------------------------------------------------------------

def plot_pipeline_stages(image, filename, path, params):
    """Figure 1: the 6 pipeline stages for one photo, in a 2 x 3 grid."""
    result = scan_document(image, keep_stages=True, **params)
    titles = {
        "1_gray": "1. Grayscale (500 px copy)",
        "2_blurred": "2. Gaussian blur 5x5",
        "3_edges": "3. Canny edges + closing" if result.method != "otsu" else "3. Otsu mask + closing",
        "4_contours": "4. Contours + detected page",
        "5_warped": "5. Perspective warp (full res)",
        "6_thresholded": "6. Adaptive threshold",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    for ax, (key, title) in zip(axes.ravel(), titles.items()):
        stage = result.stages.get(key)
        if stage is None:
            ax.axis("off")
            ax.set_title(f"{title}\n(not available: {result.method})", fontsize=9)
            continue
        show(ax, stage, title)
    fig.suptitle(f"Pipeline stages for {filename} (method: {result.method})", color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_before_after(records, path, count=6):
    """Figure 2: before/after pairs for up to `count` successful scans (3 pairs per row)."""
    chosen = [r for r in records if r["success"] is True][:count]
    if not chosen:
        return _placeholder(path, "No successful scans to show")
    rows = math.ceil(len(chosen) / 3)
    fig, axes = plt.subplots(rows, 6, figsize=(15, 4.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, record in enumerate(chosen):
        r, c = divmod(i, 3)
        before, after = axes[r][2 * c], axes[r][2 * c + 1]
        before.axis("on")
        after.axis("on")
        show(before, record["thumb_detected"], f"{record['filename']}\nbefore")
        show(after, record["thumb_scan"], f"after (IoU {record['iou']:.2f})")
    fig.suptitle("Before / after: original photo with detected outline, and final scan", color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_failure_gallery(records, path, iou_threshold):
    """Figure 3: every failure, detected quad in red and ground truth in green."""
    failures = [r for r in records if r["success"] is False]
    if not failures:
        return _placeholder(path, f"No failures at IoU >= {iou_threshold:.2f}")
    cols = min(4, len(failures))
    rows = math.ceil(len(failures) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, record in zip(axes.ravel(), failures):
        ax.axis("on")
        show(ax, record["thumb_failure"],
             f"{record['filename']}\n{record['method']}, IoU {record['iou']:.2f}")
    handles = [plt.Line2D([], [], color="#e62828", lw=2, label="detected"),
               plt.Line2D([], [], color="#28aa28", lw=2, label="ground truth")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Failure cases ({len(failures)})", color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_success_by_condition(df, path):
    """Figure 4: success rate for each value of each condition (4 small charts, same 0-100% axis)."""
    annotated = df[df["success"].notna()]
    overall = annotated["success"].astype(bool).mean() if len(annotated) else float("nan")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharex=True)
    for ax, column in zip(axes, CONDITION_COLUMNS):
        table = group_table(df, column, CONDITION_VALUES[column])
        ax.set_title(column.capitalize(), color=INK, loc="left")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        if table.empty:
            ax.text(0.5, 0.5, "not filled in\nconditions.csv", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
            ax.set_yticks([])
            continue
        labels = table[column].tolist()[::-1]
        rates = table["success_rate"].tolist()[::-1]
        counts = table["n"].tolist()[::-1]
        ax.barh(labels, rates, color=BLUE, height=0.6)
        for y, (rate, count) in enumerate(zip(rates, counts)):
            ax.text(rate + 0.02, y, f"{rate:.0%}  (n={count})", va="center", fontsize=8.5, color=INK)
        ax.tick_params(axis="y", length=0)
        if not math.isnan(overall):
            ax.axvline(overall, color=MUTED, lw=1, ls="--")
    fig.suptitle("Success rate by condition (dashed line = overall rate)", color=INK, x=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_iou_distribution(df, path, iou_threshold):
    """Figure 5: IoU of every annotated photo, sorted, with the success threshold."""
    data = df[df["success"].notna()].sort_values("iou")
    if data.empty:
        return _placeholder(path, "No annotated photos")
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(data) + 2), 3.8))
    colors = [BLUE if s else ORANGE for s in data["success"]]
    ax.bar(range(len(data)), data["iou"], color=colors, width=0.7)
    ax.axhline(iou_threshold, color=MUTED, ls="--", lw=1)
    ax.text(len(data) - 0.5, iou_threshold + 0.015, f"success threshold {iou_threshold:.2f}",
            ha="right", fontsize=8.5, color=MUTED)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["filename"], rotation=70, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("IoU with ground truth")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="success"),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="failure")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    ax.set_title("IoU per photo (sorted)", color=INK, loc="left")
    fig.savefig(path)
    plt.close(fig)


def plot_ocr_cer(df, path):
    """Figure 6 (optional): OCR character error rate on the raw photo vs the scanned output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(df) + 2), 3.8))
    ax.bar(x - 0.18, df["cer_raw"], width=0.34, color=ORANGE, label="raw photo")
    ax.bar(x + 0.18, df["cer_scan"].fillna(0), width=0.34, color=BLUE, label="scanned output")
    ax.set_xticks(x)
    ax.set_xticklabels(df["filename"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Character error rate (lower is better)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("OCR accuracy: raw photo vs scanned output", color=INK, loc="left")
    fig.savefig(path)
    plt.close(fig)


def _placeholder(path, message):
    """Save a small figure with just a message (so every expected figure file exists)."""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=12)
    fig.savefig(path)
    plt.close(fig)


# --- Main evaluation loop -----------------------------------------------------------

def evaluate(images_dir, gt_path, conditions_path, out_dir, iou_threshold, params,
             sample_name=None, overwrite_analysis=False):
    """Run the scanner on every photo, compare with ground truth and write all outputs."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(images_dir)
    if not files:
        print(f"No images found in {images_dir}. Put your photos there first.")
        return None
    truth = read_ground_truth(gt_path)
    conditions = read_conditions(conditions_path)
    if not truth:
        print(f"WARNING: no ground truth in {gt_path}. Run annotate.py first; "
              "IoU and success rates will be empty.")
    if conditions.empty:
        print(f"WARNING: {conditions_path} missing or empty. Create it with "
              "'python annotate.py --conditions-template' and fill it in.")

    records = []
    for path in files:
        image = load_image(path)
        if image is None:
            print(f"[skip] {path.name}: unreadable")
            continue
        result = scan_document(image, **params)
        gt = truth.get(path.name)
        cond = conditions.loc[path.name].to_dict() if path.name in conditions.index else {}

        if gt is None:
            iou, success = float("nan"), None
        else:
            iou = polygon_iou(result.corners, gt) if result.success else 0.0
            success = bool(iou >= iou_threshold)

        # Keep only small thumbnails in memory (full-size photos would use GBs).
        thumb, scale = thumbnail(image)
        detected_thumb = draw_outline(thumb, None if result.corners is None else result.corners * scale,
                                      GREEN_BGR)
        failure_thumb = thumb.copy()
        if gt is not None:
            failure_thumb = draw_outline(failure_thumb, gt * scale, GREEN_BGR)
        if result.corners is not None:
            failure_thumb = draw_outline(failure_thumb, result.corners * scale, RED_BGR)
        scan_image = result.scanned if result.scanned is not None else result.warped

        record = {
            "filename": path.name,
            "method": result.method,
            "iou": iou,
            "success": success,
            **{c: cond.get(c, "") for c in CONDITION_COLUMNS},
            "time_ms": round(result.time_ms, 1),
            "likely_cause": "",
            "thumb_detected": detected_thumb,
            "thumb_failure": failure_thumb,
            "thumb_scan": thumbnail(scan_image)[0] if scan_image is not None else None,
        }
        if success is False:
            record["likely_cause"] = likely_cause(result.method, iou, result.corners, gt, cond)
        records.append(record)

        status = "n/a" if success is None else ("OK  " if success else "FAIL")
        iou_text = "  -  " if gt is None else f"{iou:.3f}"
        print(f"{status:>4}  {path.name:<28} method={result.method:<12} IoU={iou_text}  "
              f"{result.time_ms:6.0f} ms")

    table_columns = ["filename", "method", "iou", "success", *CONDITION_COLUMNS, "time_ms",
                     "likely_cause"]
    df = pd.DataFrame([{k: r[k] for k in table_columns} for r in records])
    df.to_csv(out_dir / "results.csv", index=False, float_format="%.4f")

    write_summary(df, out_dir, iou_threshold)

    analysis_path = out_dir / "failure_analysis.md"
    if analysis_path.exists() and not overwrite_analysis:
        analysis_path = out_dir / "failure_analysis_auto.md"
        print(f"\nfailure_analysis.md already exists (keeping your edits); "
              f"new auto version written to {analysis_path}. Use --overwrite-analysis to replace it.")
    write_failure_analysis(df, analysis_path, iou_threshold)

    # Figure 1 uses one photo: the one given with --sample, else the first success.
    successes = [r for r in records if r["success"]] or [r for r in records if r["method"] != "failed"]
    sample = sample_name or (successes[0]["filename"] if successes else records[0]["filename"])
    plot_pipeline_stages(load_image(Path(images_dir) / sample), sample,
                         fig_dir / "fig1_pipeline_stages.png", params)
    plot_before_after(records, fig_dir / "fig2_before_after.png")
    plot_failure_gallery(records, fig_dir / "fig3_failure_gallery.png", iou_threshold)
    plot_success_by_condition(df, fig_dir / "fig4_success_by_condition.png")
    plot_iou_distribution(df, fig_dir / "fig5_iou_distribution.png", iou_threshold)
    print(f"\nWrote {out_dir}/results.csv, summary.md, {analysis_path.name} and figures in {fig_dir}/")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scanner against hand-annotated corners.")
    parser.add_argument("--images", default="data/raw", help="folder of test photos")
    parser.add_argument("--gt", default="data/ground_truth.csv", help="ground-truth corners CSV")
    parser.add_argument("--conditions", default="data/conditions.csv", help="conditions CSV")
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--iou", type=float, default=0.90, help="IoU needed to count as success")
    parser.add_argument("--sample", default=None, help="photo to use for the pipeline-stages figure")
    parser.add_argument("--canny-low", type=int, default=75)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--c", type=float, default=10)
    parser.add_argument("--overwrite-analysis", action="store_true",
                        help="replace an existing (possibly hand-edited) failure_analysis.md")
    parser.add_argument("--ocr", action="store_true",
                        help="also run the OCR CER evaluation (needs Tesseract + truth files)")
    parser.add_argument("--ocr-truth", default=None,
                        help="folder of OCR truth .txt files (default: data/ocr_truth for data/raw)")
    args = parser.parse_args()

    params = {"canny_low": args.canny_low, "canny_high": args.canny_high,
              "block_size": args.block_size, "c": args.c}
    df = evaluate(args.images, args.gt, args.conditions, args.out, args.iou, params,
                  args.sample, args.overwrite_analysis)
    if df is None:
        return 1
    if args.ocr:
        from ocr import evaluate_ocr
        truth_dir = args.ocr_truth or (Path(args.images).parent / "ocr_truth"
                                       if Path(args.images).name == "raw"
                                       else Path(args.images) / "ocr_truth")
        evaluate_ocr(args.images, truth_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

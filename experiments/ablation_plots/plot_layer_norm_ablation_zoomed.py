import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "checkpoints" / "tinystories_target_135_v2_first500" / "metrics.jsonl"
NO_RMSNORM_PATH = ROOT / "checkpoints" / "no_rmsnorm_ablation_dense_500" / "metrics.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "ablation_plots" / "layer_norm_ablation_zoomed.svg"


def load_metrics(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda row: row["step"])
    return rows


def main():
    baseline = load_metrics(BASELINE_PATH)
    no_rmsnorm = load_metrics(NO_RMSNORM_PATH)

    baseline_steps = [row["step"] for row in baseline]
    baseline_vals = [row["val_loss"] for row in baseline]
    no_steps = [row["step"] for row in no_rmsnorm]
    no_vals = [row["val_loss"] for row in no_rmsnorm]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    width = 1000
    height = 620
    left = 90
    right = 40
    top = 55
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min, x_max = 0.0, 520.0
    y_min, y_max = 1.5, 18.5

    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - y) / (y_max - y_min) * plot_h

    def polyline(points):
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)

    baseline_points = list(zip(baseline_steps, baseline_vals))
    no_points = list(zip(no_steps, no_vals))

    x_ticks = [0, 100, 200, 300, 400, 500]
    y_ticks = [2, 4, 6, 8, 10, 12, 14, 16, 18]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')

    svg.append(f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="22" font-family="Arial">Layer Norm Ablation (Zoomed Early Training)</text>')

    for xt in x_ticks:
        x = sx(xt)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e6e6e6" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{height - 45}" text-anchor="middle" font-size="13" font-family="Arial" fill="#444">{xt}</text>')
    for yt in y_ticks:
        y = sy(yt)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="13" font-family="Arial" fill="#444">{yt}</text>')

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>')
    svg.append(f'<text x="{width/2:.0f}" y="{height - 10}" text-anchor="middle" font-size="16" font-family="Arial">Step</text>')
    svg.append(f'<text x="28" y="{height/2:.0f}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 28 {height/2:.0f})">Validation Loss</text>')

    svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{polyline(baseline_points)}"/>')
    svg.append(f'<polyline fill="none" stroke="#d62728" stroke-width="3" points="{polyline(no_points)}"/>')

    for x, y in baseline_points:
        svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="#1f77b4"/>')
    for x, y in no_points:
        svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="#d62728"/>')

    legend_x = 650
    legend_y = 70
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="290" height="64" rx="8" fill="#ffffff" stroke="#cccccc"/>')
    svg.append(f'<line x1="{legend_x + 18}" y1="{legend_y + 22}" x2="{legend_x + 58}" y2="{legend_y + 22}" stroke="#1f77b4" stroke-width="3"/>')
    svg.append(f'<circle cx="{legend_x + 38}" cy="{legend_y + 22}" r="4.5" fill="#1f77b4"/>')
    svg.append(f'<text x="{legend_x + 70}" y="{legend_y + 27}" font-size="14" font-family="Arial">Baseline (with RMSNorm)</text>')
    svg.append(f'<line x1="{legend_x + 18}" y1="{legend_y + 47}" x2="{legend_x + 58}" y2="{legend_y + 47}" stroke="#d62728" stroke-width="3"/>')
    svg.append(f'<circle cx="{legend_x + 38}" cy="{legend_y + 47}" r="4.5" fill="#d62728"/>')
    svg.append(f'<text x="{legend_x + 70}" y="{legend_y + 52}" font-size="14" font-family="Arial">No RMSNorm</text>')

    no_annot_x = 450
    no_annot_y = 220
    svg.append(f'<rect x="{no_annot_x}" y="{no_annot_y}" width="360" height="88" rx="8" fill="#fff5f5" stroke="#d62728"/>')
    no_text = [
        "No more points after step 100:",
        "training became numerically unstable,",
        "and later evaluation overflowed",
        "when computing perplexity.",
    ]
    for i, line in enumerate(no_text):
        svg.append(f'<text x="{no_annot_x + 14}" y="{no_annot_y + 24 + i*18}" font-size="14" font-family="Arial" fill="#7f0000">{line}</text>')
    svg.append(
        f'<line x1="{no_annot_x}" y1="{no_annot_y + 42}" x2="{sx(100):.2f}" y2="{sy(no_vals[-1]):.2f}" stroke="#d62728" stroke-width="2"/>'
    )

    base_annot_x = 430
    base_annot_y = 370
    svg.append(f'<rect x="{base_annot_x}" y="{base_annot_y}" width="395" height="68" rx="8" fill="#f4f8ff" stroke="#1f77b4"/>')
    base_text = [
        "Baseline is only logged at steps 0 and 500",
        "because this reference run used eval_interval=500.",
    ]
    for i, line in enumerate(base_text):
        svg.append(f'<text x="{base_annot_x + 14}" y="{base_annot_y + 24 + i*19}" font-size="14" font-family="Arial" fill="#0b3d91">{line}</text>')
    svg.append(
        f'<line x1="{base_annot_x}" y1="{base_annot_y + 32}" x2="{sx(500):.2f}" y2="{sy(baseline_vals[-1]):.2f}" stroke="#1f77b4" stroke-width="2"/>'
    )

    svg.append("</svg>")
    OUTPUT_PATH.write_text("\n".join(svg), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

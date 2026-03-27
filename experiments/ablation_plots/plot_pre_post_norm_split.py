import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRE_PATH = ROOT / "checkpoints" / "tinystories_target_135_v2" / "metrics.jsonl"
POST_PATH = ROOT / "checkpoints" / "post_norm_ablation_3k" / "metrics.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "ablation_plots" / "pre_vs_post_norm_split.svg"


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


def build_panel(points_pre, points_post, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h, title):
    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - y) / (y_max - y_min) * plot_h

    def polyline(points):
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points if x_min <= x <= x_max)

    svg = []
    x_ticks = [x_min, x_max] if x_max - x_min <= 500 else [x_min, x_min + 500, x_max]
    if x_min == 0 and x_max == 500:
        x_ticks = [0, 100, 200, 300, 400, 500]
    elif x_min == 500 and x_max == 3000:
        x_ticks = [500, 1000, 1500, 2000, 2500, 3000]

    y_span = y_max - y_min
    if y_span > 5:
        y_ticks = [2, 4, 6, 8]
    else:
        y_ticks = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    for xt in x_ticks:
        x = sx(xt)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#ececec" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="12" font-family="Arial" fill="#444">{int(xt)}</text>')
    for yt in y_ticks:
        if y_min <= yt <= y_max:
            y = sy(yt)
            label = f"{yt:.1f}" if y_span <= 1 else f"{int(yt)}"
            svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ececec" stroke-width="1"/>')
            svg.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#444">{label}</text>')

    svg.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.3"/>')
    svg.append(f'<text x="{left + plot_w/2:.2f}" y="{top - 12}" text-anchor="middle" font-size="16" font-family="Arial">{title}</text>')

    svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{polyline(points_pre)}"/>')
    svg.append(f'<polyline fill="none" stroke="#d62728" stroke-width="3" points="{polyline(points_post)}"/>')

    for x, y in points_pre:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.5" fill="#1f77b4"/>')
    for x, y in points_post:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.5" fill="#d62728"/>')

    return svg


def main():
    pre_rows = load_metrics(PRE_PATH)
    post_rows = load_metrics(POST_PATH)

    pre_points = [(row["step"], row["val_loss"]) for row in pre_rows if row["step"] <= 3000]
    post_points = [(row["step"], row["val_loss"]) for row in post_rows]

    width = 1180
    height = 560
    margin_left = 70
    margin_right = 35
    top = 90
    bottom = 80
    gap = 55
    plot_w = (width - margin_left - margin_right - gap) / 2
    plot_h = height - top - bottom

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{width/2:.0f}" y="34" text-anchor="middle" font-size="24" font-family="Arial">Pre-norm vs Post-norm Transformer</text>')

    legend_x = 385
    legend_y = 52
    svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 40}" y2="{legend_y}" stroke="#1f77b4" stroke-width="3"/>')
    svg.append(f'<circle cx="{legend_x + 20}" cy="{legend_y}" r="4.5" fill="#1f77b4"/>')
    svg.append(f'<text x="{legend_x + 52}" y="{legend_y + 5}" font-size="14" font-family="Arial">Pre-norm baseline</text>')
    svg.append(f'<line x1="{legend_x + 210}" y1="{legend_y}" x2="{legend_x + 250}" y2="{legend_y}" stroke="#d62728" stroke-width="3"/>')
    svg.append(f'<circle cx="{legend_x + 230}" cy="{legend_y}" r="4.5" fill="#d62728"/>')
    svg.append(f'<text x="{legend_x + 262}" y="{legend_y + 5}" font-size="14" font-family="Arial">Post-norm</text>')

    left1 = margin_left
    left2 = margin_left + plot_w + gap

    svg.extend(build_panel(pre_points, post_points, 0, 500, 1.8, 9.5, left1, top, plot_w, plot_h, "0-500 step"))
    svg.extend(build_panel(pre_points, post_points, 500, 3000, 1.48, 2.08, left2, top, plot_w, plot_h, "500-3000 step"))

    svg.append(f'<text x="{width/2:.0f}" y="{height - 18}" text-anchor="middle" font-size="16" font-family="Arial">Step</text>')
    svg.append(f'<text x="25" y="{height/2:.0f}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 25 {height/2:.0f})">Validation Loss</text>')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(svg + ["</svg>"]), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

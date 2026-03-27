import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metric curves from multiple checkpoint directories.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form checkpoint_dir=label",
    )
    parser.add_argument(
        "--metric",
        choices=("val_loss", "train_loss"),
        default="val_loss",
    )
    parser.add_argument(
        "--x-axis",
        choices=("step", "elapsed_seconds"),
        default="step",
    )
    parser.add_argument(
        "--title",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def load_metric(run_dir: Path, metric: str, x_axis: str) -> tuple[list[float], list[float]]:
    metrics_path = run_dir / "metrics.jsonl"
    xs: list[float] = []
    values: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            xs.append(float(record[x_axis]))
            values.append(float(record[metric]))
    return xs, values


def draw_svg(
    series: list[tuple[str, list[float], list[float]]],
    metric: str,
    x_axis: str,
    title: str,
    output_path: Path,
) -> None:
    width = 1200
    height = 800
    left = 90
    right = 260
    top = 60
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    all_steps = [step for _, steps, _ in series for step in steps]
    all_values = [value for _, _, values in series for value in values if value == value]
    min_step = min(all_steps)
    max_step = max(all_steps)
    min_value = min(all_values)
    max_value = max(all_values)
    value_padding = 0.05 * (max_value - min_value if max_value > min_value else 1.0)
    min_value -= value_padding
    max_value += value_padding

    def x_scale(step: float) -> float:
        if max_step == min_step:
            return left + plot_width / 2
        return left + (step - min_step) * plot_width / (max_step - min_step)

    def y_scale(value: float) -> float:
        if max_value == min_value:
            return top + plot_height / 2
        return top + (max_value - value) * plot_height / (max_value - min_value)

    x_ticks = 8
    x_tick_values = [min_step + i * (max_step - min_step) / x_ticks for i in range(x_ticks + 1)]
    y_ticks = 7
    y_tick_values = [min_value + i * (max_value - min_value) / y_ticks for i in range(y_ticks + 1)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="24" font-family="Arial">{escape(title)}</text>',
    ]

    for tick in x_tick_values:
        x = x_scale(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e0e0e0" stroke-width="1"/>')
        tick_label = f"{tick:.1f}s" if x_axis == "elapsed_seconds" else f"{int(round(tick))}"
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-size="14" font-family="Arial">{tick_label}</text>')

    for tick in y_tick_values:
        y = y_scale(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e0e0e0" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="14" font-family="Arial">{tick:.2f}</text>')

    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black" stroke-width="2"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-size="18" font-family="Arial">{"Elapsed seconds" if x_axis == "elapsed_seconds" else "Step"}</text>',
            f'<text x="28" y="{top + plot_height / 2}" text-anchor="middle" font-size="18" font-family="Arial" transform="rotate(-90 28 {top + plot_height / 2})">{escape(metric)}</text>',
        ]
    )

    legend_x = left + plot_width + 20
    legend_y = top + 20
    parts.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-size="18" font-family="Arial">Runs</text>')

    for index, (label, steps, values) in enumerate(series):
        color = colors[index % len(colors)]
        filtered = [(step, value) for step, value in zip(steps, values) if value == value]
        points = " ".join(f"{x_scale(step):.2f},{y_scale(value):.2f}" for step, value in filtered)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        for step, value in filtered:
            parts.append(f'<circle cx="{x_scale(step):.2f}" cy="{y_scale(value):.2f}" r="3" fill="{color}"/>')
        legend_item_y = legend_y + 24 * (index + 1)
        parts.append(f'<line x1="{legend_x}" y1="{legend_item_y - 5}" x2="{legend_x + 28}" y2="{legend_item_y - 5}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{legend_item_y}" font-size="15" font-family="Arial">{escape(label)}</text>')

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    series: list[tuple[str, list[float], list[float]]] = []
    for run_spec in args.run:
        run_path_str, label = run_spec.split("=", 1)
        run_dir = Path(run_path_str)
        steps, values = load_metric(run_dir, args.metric, args.x_axis)
        series.append((label, steps, values))

    draw_svg(series, args.metric, args.x_axis, args.title, args.output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


COLORS = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "black": "#222222",
    "white_0": "#bbbbbb",
    "white_1": "#eeeeee",
    "smth": "#9467bd",
}


def load_points(result_dir):
    points = {name: {"init": [], "pend": [], "mend": []}
              for name in COLORS}
    for path in sorted(Path(result_dir).glob("q1q2r_*.pkl")):
        with path.open("rb") as stream:
            rank_points = pickle.load(stream)
        for color, values in rank_points.items():
            points.setdefault(color, {"init": [], "pend": [], "mend": []})
            for key in ("init", "pend", "mend"):
                points[color][key].extend(values.get(key, []))
    return points


def unique(values):
    return sorted(set(float(value) for value in values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plane", choices=("q1q2", "q1r"), default="q1q2")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
    points = load_points(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    all_init = []
    for values in points.values():
        all_init.extend(values["init"])
    if not all_init:
        raise RuntimeError("No classified points found")

    arr = np.asarray(all_init, dtype=float)
    if arr.shape[1] < 4:
        raise RuntimeError("Unexpected result format")
    arr = arr[:, 1:4]
    q1, q2, r = arr[:, 0], arr[:, 1], arr[:, 2]

    if args.plane == "q1q2":
        fixed_values = unique(r)
        x_label, y_label = "q1", "q2"
        x, y, fixed_index = q1, q2, 2
    else:
        fixed_values = unique(q2)
        x_label, y_label = "q1", "r"
        x, y, fixed_index = q1, r, 1

    for frame, fixed in enumerate(fixed_values):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
        fixed_array = r if args.plane == "q1q2" else q2
        for color, values in points.items():
            init = np.asarray(values["init"], dtype=float)
            if not init.size:
                continue
            init = init[:, 1:4]
            mask = np.isclose(init[:, fixed_index], fixed, rtol=1e-7, atol=1e-10)
            selected = init[mask]
            if selected.size:
                ax.scatter(selected[:, 0] if args.plane == "q1q2" else selected[:, 0],
                           selected[:, 1] if args.plane == "q1q2" else selected[:, 2],
                           s=8, c=COLORS.get(color, "#9467bd"), label=color,
                           alpha=0.7, edgecolors="none")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(False)
        ax.legend(markerscale=2)
        fig.tight_layout()
        fig.savefig(output / "{}_{:04d}.png".format(args.plane, frame))
        plt.close(fig)


if __name__ == "__main__":
    main()

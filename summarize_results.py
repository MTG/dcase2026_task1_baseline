import os
import re
import numpy as np
from collections import defaultdict


def summarize_metrics(root_dir="model_output", output_file="summary_metrics.txt"):
    """
    Scan experiment folders for results.txt files, extract metrics,
    and compute mean ± std across folds per experiment in the 'root_dir'.
    """

    pattern = re.compile(r'([a-zA-Z0-9_]+):\s*([\d.]+)%')
    metrics_data = defaultdict(lambda: defaultdict(list))

    # --- Scan files ---
    for root, _, files in os.walk(root_dir):
        if "results.txt" in files:
            file_path = os.path.join(root, "results.txt")

            # Extract experiment name (remove fold folders)
            rel_path = os.path.relpath(root, root_dir)
            parts = rel_path.split(os.sep)
            parts = [p for p in parts if not p.lower().startswith("fold")]
            exp_name = "/".join(parts) if parts else "root"

            # Read file
            with open(file_path, "r") as f:
                content = f.read()

                for match in pattern.finditer(content):
                    name = match.group(1)
                    value = float(match.group(2))
                    metrics_data[exp_name][name].append(value)

    # --- Write summary ---
    summary_path = os.path.join(root_dir, output_file)

    with open(summary_path, "w") as out:
        out.write("=== Experiment Summary ===\n\n")

        for exp_name, metrics in sorted(metrics_data.items()):
            out.write(f"{exp_name}\n")

            for metric_name, values in sorted(metrics.items()):
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    out.write(f"  {metric_name:12s}: {mean:.2f}% ± {std:.2f}%\n")
                else:
                    out.write(f"  {metric_name:12s}: No data\n")

            num_runs = len(next(iter(metrics.values()), []))
            out.write(f"  runs        : {num_runs}\n\n")

    print(f"Summary written to {summary_path}")
    return metrics_data

if __name__ == "__main__":
    summarize_metrics()
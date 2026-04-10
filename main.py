import subprocess
import sys


def run(script):
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed!")

pipeline = [
    ("Dataset", "build_dataset.py"),
    ("Training", "train_test.py"),
    ("Results", "summarize_results.py"),
]

for name, script in pipeline:
    print(f"\n------- Process: {name} -------")
    print(f"\nRunning {script}...\n")
    run(script)
    print(f"\nFinished {script}")

print("Processes completed. Check the output folders for results.")

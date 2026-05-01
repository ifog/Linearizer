"""
End-to-end runner: trains all models then runs all evaluations.
"""

import os
import sys
import subprocess
import time

# Ensure we can import from contractive_rl/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def run_script(script_name, *args):
    """Run a script in the same interpreter."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    # Use the current interpreter (resolved from environment variable if set)
    python_exec = os.environ.get("CONTRACTIVE_PYTHON", sys.executable)
    if not python_exec:
        python_exec = "python3"
    cmd = [python_exec, script_path] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Completed in {elapsed:.1f}s")
    return elapsed


def main():
    t_total = time.time()

    print("=" * 60)
    print("CONTRACTIVE LINEARIZER RL — FULL EXPERIMENT SUITE")
    print("=" * 60)

    # Step 1: Train all models
    print("\n[Step 1] Training all models...")
    run_script("train.py")

    # Step 2: Evaluate
    print("\n[Step 2] Running evaluations...")
    run_script("evaluate.py")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"COMPLETE — Total time: {elapsed:.1f}s")
    print(f"Results saved to: {os.path.join(SCRIPT_DIR, 'results')}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

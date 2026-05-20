import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_times(path: Path):
    ns = []
    dense = []
    csr = []
    mpi_compute = []
    mpi_total = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ns.append(int(row["N"]))
            dense.append(float(row["dense_sec"]))
            csr.append(float(row["csr_sec"]))
            mpi_compute.append(float(row["mpi_compute_sec"]))
            mpi_total.append(float(row["mpi_total_sec"]))

    return ns, dense, csr, mpi_compute, mpi_total


def read_speedup(path: Path):
    ns = []
    speedup = []
    efficiency = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ns.append(int(row["N"]))
            speedup.append(float(row["speedup_total"]))
            efficiency.append(float(row["efficiency_total"]))

    return ns, speedup, efficiency


def plot_times(csv_path: Path, out_path: Path):
    ns, dense, csr, mpi_compute, mpi_total = read_times(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(ns, dense, marker="o", label="Dense")
    plt.plot(ns, csr, marker="o", label="CSR")
    plt.plot(ns, mpi_compute, marker="o", label="MPI compute-only")
    plt.plot(ns, mpi_total, marker="o", label="MPI total")

    plt.xlabel("Matrix size N")
    plt.ylabel("Time (sec)")
    plt.title("Sparse Matrix Multiplication Timing")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def plot_speedup(csv_path: Path, out_path: Path):
    ns, speedup, efficiency = read_speedup(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(ns, speedup, marker="o", label="Speedup (CSR / MPI)")
    if speedup != efficiency:
        plt.plot(ns, efficiency, marker="o", label="Efficiency (Speedup / P)")

    plt.xlabel("Matrix size N")
    plt.ylabel("Value")
    plt.title("Speedup and Efficiency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    csv_path = root / "times.csv"
    out_path = root / "times.png"
    speedup_csv = root / "speedup.csv"
    speedup_out = root / "speedup.png"

    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run the C++ app first.")

    plot_times(csv_path, out_path)
    print(f"Wrote {out_path}")

    if not speedup_csv.exists():
        raise SystemExit(f"Missing {speedup_csv}. Run the C++ app first.")

    plot_speedup(speedup_csv, speedup_out)
    print(f"Wrote {speedup_out}")

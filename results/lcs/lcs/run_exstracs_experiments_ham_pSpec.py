import os
import datetime
from run_exstracs_experiments_cv import run_cv, PROJECT_ROOT


def main():
    # Exact parameters requested
    param_grid = [
        {
            "N": 1500,
            "learningIterations": 100000,
            "theta_sel": 0.8,
            "p_spec": 0.2,
            "theta_GA": 15,
            "nu": 3.0,
        }
    ]

    out_dir = os.path.join(PROJECT_ROOT, "lcs")

    # Run HAM10000 only
    ham = run_cv(
        "csv_outputs/ham10000_wavelet_normalized.csv",
        "HAM10000",
        param_grid,
        n_splits=5,
        out_dir=out_dir,
    )

    # Save single-dataset summary with a distinguishable filename
    final_out = os.path.join(
        out_dir,
        f"exstracs_HAM10000_pSpec0.2_thetaGA15_nu3.0_N1500_iters100000_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    ham.to_csv(final_out, index=False)
    print(f"\nDone. HAM10000 summary saved to: {final_out}")


if __name__ == "__main__":
    main()

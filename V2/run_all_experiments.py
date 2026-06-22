
"""
Run the complete experiment pipeline.

This script executes the main Jupyter notebook and verifies that the
important result CSV files are produced.

Run from the repository root:

    python run_all_experiments.py

The script intentionally keeps the notebook as the single source of truth
for the experimental pipeline. This avoids duplicating complex experiment
logic in two separate places.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

NOTEBOOK_NAME = "phishing_detection_critical_evaluation_v2.ipynb"
EXECUTED_NOTEBOOK_NAME = "phishing_detection_critical_evaluation_executed_v2.ipynb"

NOTEBOOK_PATH = REPO_ROOT / NOTEBOOK_NAME
EXECUTED_NOTEBOOK_PATH = REPO_ROOT / EXECUTED_NOTEBOOK_NAME

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


REQUIRED_INPUT_FILES = [
    REPO_ROOT / "5.urldata.csv",
    REPO_ROOT / "URLFeatureExtraction.py",
    REPO_ROOT / "feature_extraction_clean.py",
    NOTEBOOK_PATH,
]


# Some earlier notebook cells save CSVs in the repository root.
# This script copies them into results/ so that all outputs are organized.
ROOT_CSV_OUTPUTS_TO_COPY = [
    "baseline_results_original_random_split.csv",
    "all_source_models_original_style_summary.csv",
    "controlled_random_split_results.csv",
    "controlled_all_source_models_summary.csv",
    "controlled_autoencoder_source_style_summary.csv",
]


EXPECTED_RESULT_FILES = [
    # EDA outputs
    RESULTS_DIR / "eda_missing_values.csv",
    RESULTS_DIR / "eda_outlier_summary.csv",
    RESULTS_DIR / "eda_crosstab_summary.csv",
    RESULTS_DIR / "eda_correlation_method_comparison.csv",
    RESULTS_DIR / "eda_temporal_analysis_limitation.csv",

    # Repeated domain-aware validation
    RESULTS_DIR / "repeated_domain_aware_all_folds.csv",
    RESULTS_DIR / "repeated_domain_aware_summary.csv",
    RESULTS_DIR / "repeated_domain_aware_summary_for_report.csv",

    # Repeated threshold analysis
    RESULTS_DIR / "repeated_threshold_results.csv",
    RESULTS_DIR / "repeated_threshold_summary.csv",
    RESULTS_DIR / "repeated_threshold_summary_for_report.csv",

    # Realistic-prevalence projection
    RESULTS_DIR / "realistic_prevalence_projection_all_folds.csv",
    RESULTS_DIR / "realistic_prevalence_projection_summary.csv",
    RESULTS_DIR / "realistic_prevalence_projection_for_report.csv",
    RESULTS_DIR / "realistic_prevalence_projection_compact_for_report.csv",

    # Feature-ablation study
    RESULTS_DIR / "feature_ablation_feature_groups.csv",
    RESULTS_DIR / "feature_ablation_all_folds.csv",
    RESULTS_DIR / "feature_ablation_summary.csv",
    RESULTS_DIR / "feature_ablation_summary_for_report.csv",
]


EXPECTED_FIGURE_FILES = [
    FIGURES_DIR / "eda_url_depth_distribution.png",
    FIGURES_DIR / "eda_spearman_correlation_matrix.png",
    FIGURES_DIR / "realistic_prevalence_precision_projection.png",
    FIGURES_DIR / "feature_ablation_balanced_accuracy.png",
]


def check_required_inputs() -> None:
    missing_files = [
        path
        for path in REQUIRED_INPUT_FILES
        if not path.exists()
    ]

    if missing_files:
        print("Missing required input files:")
        for path in missing_files:
            print(f"  - {path.relative_to(REPO_ROOT)}")

        sys.exit(1)

    print("All required input files were found.")


def ensure_output_directories() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Output directories are ready:")
    print(f"  - {RESULTS_DIR.relative_to(REPO_ROOT)}")
    print(f"  - {FIGURES_DIR.relative_to(REPO_ROOT)}")


def execute_notebook() -> None:
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(NOTEBOOK_PATH),
        "--output",
        str(EXECUTED_NOTEBOOK_PATH),
        "--ExecutePreprocessor.timeout=7200",
        "--ExecutePreprocessor.kernel_name=python3",
    ]

    print("Executing notebook...")
    print(" ".join(command))

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )

    print(f"Executed notebook saved to: {EXECUTED_NOTEBOOK_PATH.name}")


def copy_root_csv_outputs_to_results() -> None:
    copied_count = 0

    for filename in ROOT_CSV_OUTPUTS_TO_COPY:
        source_path = REPO_ROOT / filename
        destination_path = RESULTS_DIR / filename

        if source_path.exists():
            shutil.copy2(source_path, destination_path)
            copied_count += 1
            print(f"Copied {filename} -> results/{filename}")

    print(f"Copied {copied_count} root-level CSV outputs into results/.")


def verify_expected_outputs() -> None:
    expected_files = [
        *EXPECTED_RESULT_FILES,
        *EXPECTED_FIGURE_FILES,
    ]

    missing_outputs = [
        path
        for path in expected_files
        if not path.exists()
    ]

    if missing_outputs:
        print("\nSome expected output files were not found:")
        for path in missing_outputs:
            print(f"  - {path.relative_to(REPO_ROOT)}")

        print(
            "\nThe notebook may have completed partially, or some output "
            "file names may differ from the expected names in this script."
        )

        sys.exit(1)

    print("\nAll expected output files were generated successfully.")


def main() -> None:
    print("=" * 80)
    print("Running phishing-detection experiment pipeline")
    print("=" * 80)

    check_required_inputs()
    ensure_output_directories()
    execute_notebook()
    copy_root_csv_outputs_to_results()
    verify_expected_outputs()

    print("=" * 80)
    print("Pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()


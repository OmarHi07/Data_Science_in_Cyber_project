# Phishing Website Detection — Critical Evaluation and Reproduction Study

## Final Updated Version After Feedback

The final updated version of the project is located in the `V2/` folder.

This folder contains the revised notebook, the cleaned feature-extraction helper module, the pinned requirements file, the reproducibility script, the generated result files, and the generated figures.

The main updated notebook is:

```text
V2/phishing_detection_critical_evaluation_v2.ipynb
```

The automatic reproducibility script is:

```text
V2/run_all_experiments.py
```

The notebook includes the original analysis together with additional sections added after feedback. To make the additions easy to identify, each new feedback-based section starts with a red markdown cell explaining what was added. A red markdown separator containing `###################` marks the end of the added feedback section. After that separator, the notebook continues with material that was already present in the earlier version.

The submitted report was also updated after the feedback. To make the changes easy to find, the new report sections were added near the end of the original report. In the PDF report, the feedback-based additions start on **page 43**.

## Requirements Note

The default `V2/requirements.txt` file contains the core dependencies needed to run the reproducibility pipeline and the main experiments.

It intentionally does **not** include TensorFlow because TensorFlow is a large package, takes a long time to install, and may cause compatibility issues on some Python versions. TensorFlow is only needed for rerunning the optional source-style autoencoder reproduction. The main reproducibility pipeline, repeated validation, threshold analysis, realistic-prevalence projection, feature ablation, EDA outputs, figures, and unit tests can be run without TensorFlow.

## Reproducing the Updated Results

To reproduce the updated project results, run the commands below from the repository root.

```powershell
cd V2
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python run_all_experiments.py
```

Expected final output:

```text
All expected output files were generated successfully.
Pipeline completed successfully.
```

The script executes the revised notebook, saves an executed copy, copies important CSV outputs into the `results/` directory, and verifies the expected figures and result files.

## Running Unit Tests

The project includes unit tests for the cleaned feature-extraction helper functions.

Run the tests from the repository root:

```powershell
python -m pytest -q
```

Expected result:

```text
13 passed
```

The tests cover IP-address detection, iframe detection, right-click detection, WHOIS fallback behavior, and output schema consistency.


## Project Description

This repository contains the final project for the course **Cybersecurity for Data Science**.

The project critically evaluates the tutorial-style GitHub project **“Phishing Website Detection by Machine Learning Techniques”** by Shreya Gopal Sundari.

The selected source proposes a supervised machine-learning pipeline for distinguishing phishing websites from legitimate websites using features extracted from URLs, domain information, and webpage behavior.

The goals of this reproduction study are:

1. Reproduce the main model-training results reported by the original source.
2. Evaluate whether the reported conclusions are supported by the supplied dataset.
3. Analyze the feature-engineering process.
4. Examine the reproducibility of the standalone feature-extraction script.
5. Evaluate model behavior under stricter experimental settings, including deduplicated and domain-aware splits.
6. Analyze the trade-off between False Positives and False Negatives.

## Selected Tutorial and Original Repository

The approved selected source is the following tutorial-style GitHub repository:

https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/

The selected tutorial and the original GitHub repository are the same source.

## Dataset Sources

The processed dataset used in this study is stored in:

```text
5.urldata.csv
```

It contains 10,000 processed URL observations:

* 5,000 phishing observations collected from PhishTank:
  https://www.phishtank.net/

* 5,000 legitimate observations selected from the University of New Brunswick ISCX-URL2016 dataset:
  https://www.unb.ca/cic/datasets/url-2016.html

The prepared processed CSV file is included in this repository so that the notebook can be executed directly.

## Repository Contents

```text
README.md
requirements.txt
phishing_detection_critical_evaluation.ipynb
Cyber_project_report.pdf
5.urldata.csv
URLFeatureExtraction.py
results/
```

### Main Files

| File                                                | Description                                                   |
| --------------------------------------------------- | ------------------------------------------------------------- |
| `Cyber_project_report.pdf`                          | Final written report                                          |
| `phishing_detection_critical_evaluation.ipynb`      | Complete analysis notebook                                    |
| `5.urldata.csv`                                     | Processed dataset used for model training and evaluation      |
| `URLFeatureExtraction.py`                           | Supporting feature-extraction script from the original source |
| `requirements.txt`                                  | Required Python packages                                      |
| `results/`                                          | Exported experiment tables                                    |
                                      

## Models Evaluated

The notebook trains and compares the following supervised classifiers:

* Decision Tree
* Random Forest
* XGBoost

The notebook also evaluates:

* An original-style random split.
* A deterministic random split with fixed seeds.
* A deduplicated random split.
* A domain-aware split.
* A full 16-feature XGBoost model.
* A lightweight URL-only XGBoost model.
* A validation-selected classification threshold.
* False Positive and False Negative patterns.

## Execution Instructions

### Option 1 — Google Colab

1. Download or clone this repository.
2. Open `phishing_detection_critical_evaluation.ipynb` in Google Colab.
3. Upload the following files into the Colab working directory if they are not already available:

   * `5.urldata.csv`
   * `URLFeatureExtraction.py`
4. Run the notebook cells from top to bottom.

### Option 2 — Local Execution

Clone the repository:

```bash
git clone https://github.com/OmarHi07/Data_Science_in_Cyber_project.git
cd Data_Science_in_Cyber_project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
phishing_detection_critical_evaluation.ipynb
```

Then run the notebook cells from top to bottom.

## Running Unit Tests

The project includes unit tests for the cleaned feature-extraction helper functions.

To run the tests from the repository root:

```bash
python -m pytest -q

## Main Findings

The original source reports that XGBoost achieves approximately 86.4% testing accuracy. This result was reproduced approximately: the reproduced XGBoost model achieved approximately 86.3% accuracy under an original-style random split.

However, the analysis identified substantial duplication and train-test overlap in the processed dataset. After removing duplicate rows and evaluating previously unseen domains, accuracy alone became misleading.

A domain-aware experiment and validation-based threshold analysis showed that model usefulness depends strongly on the trade-off between phishing recall and legitimate-site specificity.

The final conclusion is that the selected source is a useful educational baseline, but its dataset construction, evaluation methodology, and standalone feature-extraction script require improvement before the proposed pipeline could be recommended for real-world deployment.

## Report

The complete report is available here:

```text
Cyber_project_report.pdf
```

## Author

Omar Hijab

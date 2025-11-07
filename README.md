# Laboratory 1 for the AIC Course

![Polito Logo](resources/logo_polito.jpg)

This repository contains all the materials, scripts, and documentation for Laboratory 1 of the **AI and Cybersecurity** course.

## Overview

This lab builds a complete intrusion detection pipeline on a curated subset of the CICIDS2017 dataset using **Feed Forward Neural Networks (FFNN)** in PyTorch.  

The laboratory is structured into six progressive tasks that comprehensively cover the intrusion detection pipeline:
- Task 1: Data cleaning, stratified splits, outlier inspection, scaling comparison (Standard vs Robust).
- Task 2: Shallow FFNN (single hidden layer) with neuron sweep and activation (Linear vs ReLU).
- Task 3: Feature bias analysis (Destination Port), port substitution experiment, feature removal impact.
- Task 4: Class imbalance mitigation via class‐weighted CrossEntropy.
- Task 5: Deep architectures, batch size impact, optimizer comparison (SGD / Momentum / AdamW).
- Task 6: Overfitting and regularization (Dropout, BatchNorm, Weight Decay) on deeper models.

## Repository Structure

```
Laboratory1/
├── lab/            # Data, notebooks and support material
├── report/         # LaTeX source files for the lab report
├── resources/      # Additional resources (e.g., links, PDFs, images)
└── README.md       # This file
```

> [!NOTE]
> The detailed lab report, including all experimental results and analysis, can be found [here](report/Laboratory1-report.pdf).

## Lab Objectives & Requirements

### Objectives:

1. Understand preprocessing choices (scaling, outlier retention).
2. Evaluate architectural depth vs minority class detection.
3. Quantify bias induced by a single feature (Destination Port).
4. Mitigate class imbalance using weighted loss.
5. Compare optimizers and batch sizes for convergence/generalization.
6. Assess regularization techniques on tabular intrusion data.

### Requirements:

- Python 3.10+
- PyTorch, scikit-learn, numpy, pandas, matplotlib, seaborn
- Dataset file: `lab/data/dataset_lab_1.csv`

## Quick Start

1. Clone:
   ```
   git clone <repo_url>
   cd Laboratory1
   ```

2. Create environment (example with venv):
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (Create `requirements.txt` if missing; minimal list: torch torchvision torchaudio scikit-learn pandas numpy seaborn matplotlib)

3. Run notebook:
   ```
   jupyter notebook lab/notebooks/Lab1_FFNN.ipynb
   ```

4. Results (plots, metrics) saved under `lab/results/images/<task>_plots/`.

## Data

Place `dataset_lab_1.csv` in `lab/data/`.  
No automatic download is performed (course-provided subset).

## Reproducing Experiments
- Set random seed (already fixed to 42 in notebook).
- To switch scaler: change `X_train_use = X_train_std` to robust variant.
- To rerun port bias test: execute Task 3 cells after initial training.

## Results Summary (High-level)

- Best shallow (ReLU, 64 neurons) balanced macro F1.
- Deep 3-layer [32,16,8] + AdamW gave strong trade-off.
- Weight decay (1e-4) sufficed; heavy Dropout/BatchNorm harmed minority recall.
- Port feature induced spurious correlation—removal reduced PortScan shortcuts.

## Authors

| Name              | GitHub                                                                                                               | LinkedIn                                                                                                                                  | Email                                                                                                            |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Andrea Botticella | [![GitHub](https://img.shields.io/badge/GitHub-Profile-informational?logo=github)](https://github.com/Botti01)       | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/andrea-botticella-353169293/) | [![Email](https://img.shields.io/badge/Email-Send-blue?logo=gmail)](mailto:andrea.botticella@studenti.polito.it) |
| Elia Innocenti    | [![GitHub](https://img.shields.io/badge/GitHub-Profile-informational?logo=github)](https://github.com/eliainnocenti) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/eliainnocenti/)               | [![Email](https://img.shields.io/badge/Email-Send-blue?logo=gmail)](mailto:elia.innocenti@studenti.polito.it)    |
| Simone Romano     | [![GitHub](https://img.shields.io/badge/GitHub-Profile-informational?logo=github)](https://github.com/sroman0)       | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/simone-romano-383277307/)     | [![Email](https://img.shields.io/badge/Email-Send-blue?logo=gmail)](mailto:simone.romano@studenti.polito.it)     |

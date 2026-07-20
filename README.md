# Group06 — NinaPro DB2 sEMG Gesture Recognition

## Project Information

- **Course:** CSE475 — Machine Learning
- **Semester:** Summer 2026
- **Group:** Group06
- **Track:** Track 1 — Graph Neural Network on Table-Style Data
- **Dataset:** NinaPro DB2
- **Application Area:** Surface electromyography (sEMG)-based hand gesture recognition
- **Problem Type:** Multiclass classification

## Group Members

| Student ID | Name |
|---|---|
| 2023-1-60-071 | Shawna Akter |
| 2023-1-60-019 | Mustari Zaman |
| 2023-1-60-207 | Mahfuz Uddin Ahmed |

## Project Overview

This project investigates the NinaPro DB2 multichannel sEMG dataset for hand gesture recognition. The work follows the Track 1 workflow: exploratory data analysis and related-work review in Task 1, machine-learning baselines and a proposed GNN in Task 2, and model improvement, ablation, explainability, cross-validation, and significance testing in Task 3.

## Dataset

- **Official source:** NinaPro DB2
- **Participants:** 40 intact subjects
- **Active gesture classes:** 49
- **Rest label:** 0, analysed separately
- **sEMG channels:** 12
- **Sampling frequency:** 2,000 Hz
- **Preferred labels:** `restimulus` and `rerepetition`
- **Observed repetition values:** 1–6

## Repository Structure

```text
Group06_NinaProDB2_CSE475/
├── README.md
├── code/
│   ├── task1/
│   │   └── Group06_NinaProDB2_task1_eda.ipynb
│   ├── task2/
│   └── task3/
├── report/
│   ├── task1/
│   ├── task2/
│   └── task3/
├── related_work/
│   ├── Group06_NinaProDB2_related_work_table.pdf
│   └── papers/
└── models/
```

## Task 1

### Exploratory Data Analysis

The Task 1 notebook includes:

- Dataset summary and metadata inspection
- Raw-signal and feature-level descriptive statistics
- Missing, infinite, duplicate, flat-channel, clipping, discontinuity, and label-quality checks
- Active-segment extraction and 200 ms windowing with 50% overlap
- MAV, RMS, variance, waveform length, zero crossing, and slope sign change features
- Class balance by window count and exact active duration
- Raw waveform and rest-versus-active comparisons
- Gesture-, subject-, channel-, and repetition-wise feature distributions
- Electrode-to-electrode and feature-to-feature correlation analysis
- PCA and t-SNE views coloured by gesture and subject
- Interactive Plotly visualizations
- Evidence-based EDA findings and limitations

### Current EDA Results

- All 40 subjects and all 49 active gesture classes are included.
- The processed data contain **11,760 active gesture segments**.
- The final window-level table contains **565,608 windows**.
- Each window contains **72 numerical sEMG features** extracted from 12 channels.
- Refined repetition values were verified as **1–6**.
- The final feature table contains no missing or infinite values and no fully duplicated rows.
- Subject-specific variation is visible in the feature distributions and dimensionality-reduction views.
- Electrode and feature correlations indicate meaningful channel relationships and potentially redundant feature pairs.

### Related Work

Task 1 also includes a structured related-work review of recent sEMG gesture-recognition studies. The table covers:

- Title
- Authors
- Year
- Dataset
- Application area
- Method/model
- GNN type
- Metrics
- Strengths
- Limitations
- Research gap
- Relation to this project

The reviewed literature is used to define the research gap for a subject-independent, EDA-guided GNN on the full NinaPro DB2 dataset.

## How to Run

1. Open `code/task1/Group06_NinaProDB2_task1_eda.ipynb` in Kaggle.
2. Attach the NinaPro DB2 data or enable Internet access according to the notebook configuration.
3. Check the configuration cell before execution.
4. Run the notebook from top to bottom.
5. Generated tables, figures, interactive plots, and checkpoints will be saved in the configured output directory.

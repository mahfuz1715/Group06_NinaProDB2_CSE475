# Graph-Based sEMG Gesture Recognition with NinaPro DB2

A CSE475 Machine Learning course project on **surface electromyography (sEMG) hand-gesture recognition** using the **NinaPro Database 2 (DB2)**.

The project starts with dataset understanding and classical machine-learning baselines, then moves to a graph-based model where the 12 sEMG electrodes are treated as connected nodes. The final stage focuses on controlled ablation, cross-validation, statistical testing, and model explainability rather than reporting a single score in isolation.

---

## Project at a Glance

| Item | Details |
|---|---|
| Course | CSE475 — Machine Learning |
| Group | Group 06 |
| Dataset | NinaPro Database 2 (DB2) |
| Exercise | Exercise B / E1 |
| Subjects | 40 |
| sEMG channels | 12 |
| Active gesture classes | 17 |
| Sampling rate | 2000 Hz |
| Window | 200 ms = 400 samples |
| Overlap | 50% |
| Train repetitions | 1, 3, 4, 6 |
| Test repetitions | 2, 5 |

**Dataset:** [NinaPro DB2 on Kaggle](https://www.kaggle.com/datasets/quddusikashaf/ninapro-db2)

The same instructor-approved repetition-wise protocol is maintained throughout the project. Test repetitions remain separate from training, and all data-dependent preprocessing is fitted on the training side only.

---

## Project Flow

### Task 1 — Dataset Understanding and EDA

The first stage was used to understand the structure and quality of the selected NinaPro DB2 data before modeling.

The analysis covers:

- Exercise B file and label verification for all 40 subjects
- corrected gesture and repetition labels
- gesture and repetition coverage
- class and segment balance
- raw sEMG signal behavior
- channel-wise and subject-wise activity analysis
- missing and infinite value checks
- related-work review and research-gap identification

For the training repetitions, the dataset contains **2,720 active gesture-repetition segments**, with **160 segments per gesture** at the segment level.

The main outcome of Task 1 was a clear motivation for graph learning: the 12 electrodes are not independent measurements, and their relationships can be modeled explicitly.

---

### Task 2 — Baselines and the First ST-EGAT

Task 2 first builds a strong non-graph benchmark and then introduces the proposed graph-attention model.

#### Feature Representation

Each 200 ms window is represented using:

- **15 handcrafted time/frequency descriptors per electrode**
- `15 × 12 = 180` electrode-level features
- **66 pairwise electrode-correlation features**
- **246 engineered features** before filtering
- **234 features** after variance filtering
- **top 200 ANOVA-selected features** for the final global feature branch

Preprocessing follows the sequence:

`Median Imputation → Variance Filtering → ANOVA Feature Selection → Standardization`

All preprocessing parameters are learned from the training repetitions only.

#### Baseline Models

Seven baseline models were evaluated:

- Logistic Regression
- Linear SVM
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- MLP

**XGBoost** was the strongest baseline.

#### Proposed Model

The proposed **ST-EGAT** combines raw temporal information, handcrafted electrode features, graph relationships, and the selected global feature representation.

```text
12 × 400 raw sEMG window
        │
        ├── Shared temporal CNN
        │
        ├── 15 handcrafted features per electrode
        │
        ▼
  12 electrode nodes
        │
  sample-specific top-k graph
        │
  edge-aware graph attention
        │
 attention + mean + max readout
        │
        ├──────────────┐
        │              │
        │        200-feature global MLP
        │              │
        └────── fusion ┘
               │
         17 gesture classes
```

The implemented graph branch uses **three edge-aware multi-head attention blocks**, with sample-specific connectivity built from the strongest inter-electrode correlations.

---

## Main Results

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| XGBoost baseline | 77.69% | 77.39% | 98.25% | 85.99% |
| **Task 2 First ST-EGAT** | **81.16%** | **80.98%** | 97.66% | **86.93%** |
| Task 3 Binary-Edge ST-EGAT | 80.37% | 80.20% | 97.59% | 86.38% |

The first ST-EGAT improved Macro-F1 by **3.59 percentage points** over XGBoost under the same train/test protocol.

The Task 2 model retains the highest single held-out score. Task 3 therefore does not claim a higher one-shot test score; its purpose is to test the graph design more rigorously and check whether the gain is stable and explainable.

---

## Task 3 — Ablation, Validation and Explainability

Task 3 evaluates whether the graph itself contributes useful information instead of assuming that a more complex model must be better.

Seven controlled variants were compared, including:

- Binary Edges
- Top-3 Graph
- No Graph
- Two GAT Blocks
- First ST-EGAT
- ST-EGAT+
- GraphSAGE

The selected configuration was **Binary-Edge ST-EGAT**.

| Validation result | Value |
|---|---:|
| Best epoch | 36 |
| Binary Edges Macro-F1 | 69.17% |
| No Graph Macro-F1 | 67.65% |
| Graph gain over No Graph | **+1.52 pp** |

After model selection, the chosen configuration was retrained on repetitions `1, 3, 4, 6` and evaluated on the untouched repetitions `2, 5`.

### Five-Fold Validation

| Model | Macro-F1 |
|---|---:|
| XGBoost | 65.10 ± 2.11% |
| **Binary-Edge ST-EGAT** | **67.13 ± 1.65%** |

A one-sided **Wilcoxon signed-rank test** on paired fold results produced:

`p = 0.03125`

This provides statistical support for the fold-wise Macro-F1 improvement under the project evaluation protocol.

---

## Explainability

The final checkpoint is examined using three complementary views:

- **electrode attention** — which graph nodes receive more importance
- **SHAP** — which engineered features contribute to a prediction
- **LIME** — a local explanation for an individual prediction

Both a correct and an incorrect test prediction are inspected. This is useful because a good explanation should help us understand not only when the model succeeds, but also when it is confidently wrong.

The explanations are treated as evidence of model behavior, not as causal physiological claims.

---

## Repository Structure

```text
Group06_NinaProDB2/
│
├── README.md
│
├── code/
│   ├── task1/
│   │   └── Group06_NinaProDB2_task1_eda.ipynb
│   │
│   ├── task2/
│   │   ├── Group06_NinaProDB2_task2_baselines.ipynb
│   │   └── Group06_NinaProDB2_task2_proposed_model.ipynb
│   │
│   └── task3/
│       ├── Group06_NinaProDB2_task3_improvement_ablation.ipynb
│       └── Group06_NinaProDB2_task3_explainability.ipynb
│
├── report/
│   ├── task1/
│   │   └── Group06_NinaProDB2_Task1_Report.pdf
│   ├── task2/
│   │   └── Group06_NinaProDB2_Task2_Report.pdf
│   └── task3/
│       └── Group06_NinaProDB2_Task3_Report.pdf
│
├── related_work/
│   ├── Group06_NinaProDB2_related_work_table.pdf
│   └── papers/
│
└── models/
    ├── Group06_NinaProDB2_best.pth
    └── label_map.json
```

---

## Running the Notebooks on Kaggle

1. Create a Kaggle notebook.
2. Add the [NinaPro DB2 dataset](https://www.kaggle.com/datasets/quddusikashaf/ninapro-db2) as an input.
3. Enable a GPU for the GNN notebooks.
4. Run the notebooks in task order.
5. Keep the same dataset path and repetition protocol used in the notebooks.
6. Run the Task 3 improvement/ablation notebook before explainability.
7. Add the saved final model/checkpoint as an input when running the explainability notebook.

The notebooks were developed for the Kaggle environment and use the dataset path:

```text
/kaggle/input/datasets/quddusikashaf/ninapro-db2
```

---

## Evaluation

Performance is reported using more than accuracy alone:

- Accuracy
- Macro and weighted Precision
- Macro and weighted Recall
- Macro and weighted F1
- Confusion matrix
- One-vs-rest ROC/AUC
- Precision-recall curves and PR-AUC
- Training time

Task 3 additionally includes **five-fold grouped validation**, **ablation analysis**, and **Wilcoxon significance testing**.

---

## Reproducibility Notes

The project uses a fixed repetition-wise split:

```text
Train: 1, 3, 4, 6
Test : 2, 5
```

Windows are kept within gesture and repetition boundaries. Imputation, variance filtering, feature selection, scaling, and class weighting are based on training data only. Task 3 model/epoch selection is performed using validation performance rather than the final held-out test set.

The protocol is **repetition-wise**, not subject-independent; the same subjects may appear on both sides through different repetitions.

---

## Team

| Student ID | Name |
|---|---|
| 2023-1-60-071 | Shawna Akter |
| 2023-1-60-019 | Mustari Zaman |
| 2023-1-60-207 | Mahfuz Uddin Ahmed |

**Department of Computer Science & Engineering**  
**East West University**

---

## Tools

Python, NumPy, Pandas, SciPy, scikit-learn, XGBoost, PyTorch, Matplotlib, SHAP, LIME, and Kaggle Notebooks.

---

## Note

The NinaPro DB2 dataset is not redistributed in this repository. It should be accessed from the dataset source linked above.

Results from related studies are used only as context because differences in gesture sets, sensor configurations, and evaluation protocols make direct score-to-score comparison unreliable.

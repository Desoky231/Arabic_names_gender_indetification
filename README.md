
# Gender Prediction from First Names – README
*(Notebook: `gender reveal.ipynb`)*

## 1 • Project Goal
Build and evaluate a machine‑learning pipeline that predicts **binary gender (Female / Male)** from first‑name strings **while explicitly measuring and mitigating bias**.

## 2 • Data Snapshot
| File | Rows | Notes |
|------|-----:|-------|
| `train_data.csv` | 55 921 | Labelled names (85 % Male / 15 % Female) |
| `arabic_names.csv` | 6 342 | Arabic‑script additions |
| `FreezedNames_final.csv` | 10 000 | Unlabelled names for deployment demo |

## 3 • Baseline Workflow
1. **Vectorisation** – character 1–3‑gram counts (`CountVectorizer`).
2. **Models** – Logistic Regression, Random Forest, HistGradientBoosting.
3. **Metrics** – overall accuracy + `classification_report` per gender.

### Baseline fairness check
Using **Fairlearn MetricFrame**: female recall = **0 %**, male recall ≈ 88 %.  
Equal‑opportunity (recall) gap ≈ 0.88  ➜ model is *heavily biased* toward males.

## 4 • Fairness Evaluation Pipeline
```python
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate
mf = MetricFrame(
    metrics={'accuracy': accuracy_score,
             'recall': recall_score,
             'FPR': false_positive_rate,
             'selection_rate': selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=y_test       # Female / Male
)
```
We monitor **recall gap**, **FPR gap**, **selection‑rate gap** for every model.

## 5 • Bias Mitigation Steps
| Step | Tool | Impact |
|------|------|--------|
| Re‑weight classes | `class_weight='balanced'` | Penalises female errors more |
| Oversample minority | `imblearn.RandomOverSampler` | 50 % / 50 % training balance |
| Threshold tuning | Grid search 0.30–0.70 to minimise recall gap | Increases female recall |
| Fairness dashboard | Bias‑aware leaderboard sorted by recall gap | Transparent model selection |

## 6 • Post‑Mitigation Results
| Model | Accuracy | Recall F | Recall M | Gap |
|-------|---------:|---------:|---------:|----:|
| RF + oversample + tuned thr | **0.83** | **0.76** | 0.84 | **0.08** |
| HistGB  (balanced)          | 0.81 | 0.71 | 0.83 | 0.12 |
| LogisticReg (balanced)      | 0.63 | 0.60 | 0.65 | 0.05* |

\*Lower gap but lower overall accuracy.

Equal‑opportunity gap dropped from **0.88 ➜ 0.08** (10× improvement).

## 7 • Responsible‑Use & Limitations
* Supports **only two classes**; non‑binary identities are out‑of‑scope.  
* Intended for **aggregate analytics**, not for personal gender assignment.  
* A confidence threshold outputs `"Unknown"` if model certainty ≤ 0.55.  
* Monitor drift & fairness monthly—Fairlearn dashboard provided.

## 8 • Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # includes fairlearn, imbalanced-learn
jupyter notebook                       # open `gender reveal.ipynb`
Kernel ▶ Restart & Run All
```
All fairness plots and tables regenerate in < 5 min on CPU.

## 9 • Dependencies
```
scikit-learn>=1.5
pandas>=2.2
fairlearn>=0.10
imbalanced-learn>=0.12
matplotlib, jupyter
```

## 10 • Next Road‑map
* Hyper‑parameter search with Optuna + fairness objective.  
* Character‑level transformer embeddings for better female precision.  
* Cross‑script bias audit (Latin vs Arabic).

---  
© 2025 A. El‑Desoky – MIT License.


# Gender Prediction from Names – Project README

## 1. Overview
This notebook walks through building a machine‑learning pipeline that **predicts a person’s binary gender (Female / Male) from their first name**.  
Beyond accuracy, the project **explicitly evaluates and mitigates gender bias** using the open‑source **Fairlearn** library.

## 2. Data
| File | Purpose |
|------|---------|
| `train_data.csv` | Labelled names (Arabic + English) with ground‑truth gender. |
| `arabic_names.csv` | Extra Arabic pairs; merged after cleaning. |
| `FreezedNames_final.csv` | Unlabelled names for final inference. |

Class distribution in the raw training set:

| Gender | Count | %
|--------|------:|----:|
| Male   | 47 292 | 85 % |
| Female |  8 629 | 15 % |

## 3. Baseline workflow
1. **Text vectorisation** – character 1–3‑gram counts via `CountVectorizer`.  
2. **Model candidates** – Logistic Regression, Random Forest, HistGradientBoosting.  
3. **Baseline accuracy** – RF ≈ 0.87.  
4. **Fairness check** – with Fairlearn `MetricFrame`:

```
Female recall = 0 %
Male   recall = 87 %
Equal‑opportunity gap ≈ 0.87
```

All three baselines defaulted to predicting *Male*, exposing severe bias.

## 4. Fairness evaluation methodology
```python
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate
mf = MetricFrame(
        metrics={"recall": recall_score,
                 "FPR": false_positive_rate,
                 "selection_rate": selection_rate},
        y_true = y_test,
        y_pred = y_pred,
        sensitive_features = y_test   # gender labels
)
```
We track **recall gap**, **FPR gap**, and **selection‑rate gap** (Demographic Parity).

## 5. Mitigation steps
| Step | Library / Technique | Effect |
|------|---------------------|--------|
| **Class weighting** | `class_weight='balanced'` in all estimators | forces loss to penalise minority class |
| **RandomOverSampler** | `imblearn.over_sampling` (↑ females to 50 %) | gives the learner equal representation |
| **Threshold tuning** | Grid‑search 0.30→0.70 on validation set to minimise recall gap | lifts female recall without tanking accuracy |
| **Bias‑aware leaderboard** | Stores per‑model gaps then sorts by smallest recall gap | transparent model selection |

## 6. Post‑mitigation results
| Model | Accuracy | Recall (F) | Recall (M) | Recall gap |
|-------|---------:|-----------:|-----------:|-----------:|
| Random Forest + oversample + tuned thr | **0.83** | **0.76** | 0.84 | **0.08** |
| HistGB + oversample + tuned thr        | 0.80 | 0.72 | 0.82 | 0.10 |
| Logistic Reg (best thr)                | 0.63 | 0.60 | 0.65 | 0.05 |

*Equal‑opportunity difference* reduced from **0.87 ➜ 0.08**.

## 7. Responsible‑use statement
*The model is intended for aggregated analytics and demo purposes only.*  
It recognises just two gender classes and may mis‑gender individuals or ignore non‑binary identities.  
A confidence threshold outputs **“Unknown”** when model certainty is low (≤ 0.55).

## 8. Reproducing the results
```bash
git clone <repo-url>
cd gender-name-gender-pred
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
# open `gender reveal.ipynb` and run all cells
```

## 9. Dependencies
```
pandas  >= 2.2
numpy   >= 1.26
scikit-learn >= 1.5
fairlearn >= 0.10
imbalanced-learn >= 0.12
matplotlib
jupyter
```

## 10. Next steps
* Hyper‑parameter tuning via Optuna.  
* Explore character‑level CNN / Bi‑LSTM for improved female precision.  
* Add multilingual support and evaluate bias across scripts (Latin vs Arabic).

---

© 2025 Abdelrhman El‑Desoky – Licensed MIT.

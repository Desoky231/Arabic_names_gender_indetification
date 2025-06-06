
# Gender Prediction from Names – Project README

## Overview
This Jupyter notebook demonstrates how to build a machine‑learning pipeline that **infers a person’s gender from their first name**.  
It walks through loading and cleaning three real‑world CSV datasets, engineering text features, training several classifiers, and evaluating their performance.

## Notebook
* **File:** `gender reveal.ipynb`
* **Goal:** Achieve the highest possible accuracy while keeping the workflow fully reproducible and explainable.

## Data Sources
| CSV | Purpose |
|-----|---------|
| `train_data.csv` | Core labelled set of Arabic and English names with ground‑truth gender. |
| `arabic_names.csv` | Supplemental Arabic name–gender pairs – merged after cleaning. |
| `FreezedNames_final.csv` | Unlabelled names – final target for prediction once the model is trained. |

*(All CSVs should live in the same directory as the notebook.)*

## Pipeline Steps
1. **Load & merge datasets**  
   Handle encoding, strip stray whitespace, and drop duplicates.
2. **Data cleaning**  
   Remove phone‑number artefacts and rows with null names/genders.
3. **EDA**  
   Quick sanity checks on class balance and character distributions.
4. **Feature engineering**  
   Use `sklearn.feature_extraction.text.CountVectorizer` to turn each name into character n‑gram counts (1‑ to 3‑grams).
5. **Train / validation split**  
   80 % train, 20 % test with a fixed random seed for reproducibility.
6. **Model training**  
   * Logistic Regression  
   * Random Forest Classifier  
   * HistGradientBoostingClassifier
7. **Evaluation**  
   Compute accuracy & confusion matrix on held‑out test data.
8. **Select best model**  
   Random Forest achieved **≈ 89 % accuracy**, edging out the boosting model.
9. **Batch prediction**  
   Apply the best estimator to `FreezedNames_final.csv` and write results back to disk.

## Reproducing the Results

### 1. Clone / download the repo
```bash
git clone <your‑repo‑url>
cd gender‑prediction‑names
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook
# open `gender reveal.ipynb`
```

### 4. Run all cells
The notebook is linear—select **Kernel ▸ Restart & Run All** to reproduce training, evaluation, and final predictions.

## Dependencies
```
pandas
numpy
scikit‑learn
matplotlib            # optional for plots
jupyter
```

*(Versions tested: pandas 2.2, scikit‑learn 1.5, Python 3.11.)*

## Results
| Model | Test Accuracy |
|-------|--------------:|
| Logistic Regression | ~84 % |
| HistGradientBoosting | ~88 % |
| **Random Forest** | **~89 %** |

## Next Steps
* Hyper‑parameter tuning (GridSearch / Optuna).  
* Try character embeddings + CNN/Bi‑LSTM on the names.  
* Explore gender‑neutral / ambiguous name handling.

---

© 2025 Abdelrhman El‑Desoky. Feel free to fork and adapt under the MIT license.

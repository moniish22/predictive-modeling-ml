# Predictive Modeling Using Machine Learning

A supervised-learning project that predicts whether a breast tumor is **malignant or benign**.

## Dataset
Breast Cancer Wisconsin (Diagnostic), loaded from `sklearn.datasets` (569 samples, 30 numeric features, no missing values).

## Models
- Logistic Regression (classification counterpart of Linear Regression)
- Decision Tree
- Random Forest

## Workflow
1. Load data and explore it (class balance, correlations)
2. Stratified 80/20 train/test split
3. Scaling inside a Pipeline (no data leakage)
4. Train three models
5. Evaluate: accuracy, precision, recall, F1, ROC AUC
6. Visualize: confusion matrices, ROC curves, feature importance, decision tree
7. 5-fold cross-validation and hyperparameter tuning (GridSearchCV)

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.982 | 0.986 | 0.986 | 0.986 | 0.995 |
| Decision Tree | 0.939 | 0.958 | 0.944 | 0.951 | 0.934 |
| Random Forest | 0.956 | 0.959 | 0.972 | 0.966 | 0.993 |

Logistic Regression performed best on this split. Random Forest was close behind, and the Decision Tree is the most interpretable but least accurate.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook predictive_modeling.ipynb
```
Then choose **Kernel > Restart & Run All**.

## Files
- `predictive_modeling.ipynb` - full notebook with outputs and plots
- `predictive_modeling.html` - same notebook as a web page (open in any browser)
- `requirements.txt` - dependencies

## Tech
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn

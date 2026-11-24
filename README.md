House Price Prediction Using Boston Housing Data
Supervised machine learning regression project predicting median residential home prices (`MEDV`, in `$1,000s`) from neighborhood, structural, and socioeconomic attributes in the Boston Housing dataset (`506` records, `13` features).
Project Overview
Data Collection & Exploration: Loaded the Boston Housing dataset via `fetch_openml(name="boston", version=1)` into a pandas DataFrame (`506 × 14`), verified zero missing values, and inspected continuous vs. categorical/discrete variables.
Exploratory Data Analysis (EDA): Built a reusable scatter-plot function (`plot_features_vs_target`) to visualize relationships between key predictors (`RM`, `LSTAT`, `AGE`, `CRIM`, `PTRATIO`, `DIS`) and `MEDV`.
Model Training: Split the data into an 80% training set (404 samples) and 20% test set (102 samples) (`random_state=42`) and trained a `RandomForestRegressor`.
Evaluation & Hyperparameter Tuning: Evaluated baseline test performance with RMSE and $R^2$, then performed 5-fold cross-validation using `GridSearchCV`.
Inference: Predicted median home value for an unseen sample neighborhood profile.
Model Results
Model Configuration	Key Parameters	Test RMSE ($1,000s)	Test $R^2$
Baseline Random Forest	`n_estimators=100` (default)	2.811 (`$2,811`)	0.8923
Tuned Random Forest (`GridSearchCV`)	`max_depth=10`, `min_samples_leaf=2`, `min_samples_split=2`, `n_estimators=100`	3.018 (`$3,018`)	0.8758
Key Findings
Top Predictors: `RM` (`51.28%` importance), `LSTAT` (`31.54%` importance), and `DIS` (`5.65%` importance) account for over 88% of the model's predictive power.
Sample Prediction: For a new neighborhood profile (`RM=6.5`, `LSTAT=12.0`, `CRIM=0.2`, etc.), the model predicts a median home value (`MEDV`) of `$22.33k` (`$22,327.84`).
Repository Contents
`Assignment_7_House_Price_Prediction_Using_Boston_Housing_Data.ipynb` — Complete Google Colab notebook with code, visualizations, evaluation metrics, and predictions.
`README.md` — Project summary and model evaluation results.
Tech Stack
Language: Python 3 (Google Colab)
Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

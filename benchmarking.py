import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LassoLars, LinearRegression
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.model_selection import KFold
import json
from pandas.io.json import json_normalize
from modelling.feat_engineer.feature_engineering import rgb_to_ppg, rolling_augment_dataset, engineer_features, _attach_sample_id_to_ground_truth
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif
import os


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


dataset = pd.read_csv(
    'preprocessed_data/combined_data/combined_data.csv')
labels = pd.read_csv(
    'preprocessed_data/combined_data/ground_truth_combined.csv')
print(f'Loaded dataset {dataset.shape} and labels {labels.shape}')

targets = ['spo2', 'hr']
models = [XGBRegressor, LassoLars, LinearRegression]
model_names = ['xgboost', 'lasso', 'lr']
K_vals = [2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]

results = {'aug_window': [], 'target': [], 'model': [], 'k': [], 'mse': []}
for aug in tqdm([60, 90, 120, 200, 240, 300], desc='Aug'):

    # Augment dataset
    aug_data = rolling_augment_dataset(dataset, n_frames=aug, step=20)
    aug_labels = _attach_sample_id_to_ground_truth(aug_data, labels)
    # Engineer features
    all_features = engineer_features(
        aug_data, aug_labels, target=targets, select=False)

    for target in targets:
        print(f"Evaluating {target}...")
        for K in tqdm(K_vals, desc=target):
            _X = all_features.drop(targets, axis=1).values
            _y = all_features[target].values
            kbest = SelectKBest(f_classif, k=K).fit(_X, _y)
            _X = _X[:, kbest.get_support()]
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(_X):
                X_train, X_test = _X[train_index], _X[test_index]
                y_train, y_test = _y[train_index], _y[test_index]
                for model, mname in zip(models, model_names):
                    mod = model().fit(X_train, y_train)
                    score = mse(y_test, mod.predict(X_test))
                    for key, value in {'target': target, 'model': mname, 'mse': score, 'k': K, 'aug_window': aug}.items():
                        results[key].append(value)

now = datetime.now().strftime("%m-%d-%H_%M_%S")
if not os.path.exists('results'):
    os.makedirs('results')
pd.DataFrame.from_dict(results).to_csv(
    f"results/{now}_benchmark.csv", index=False)

import pickle
import matplotlib.pyplot as plt
import numpy as np

from modelling.feat_engineer.data_processing import make_seq
from modelling.uncertainty_model.model import  evaluate
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import datetime

if __name__ == '__main__':
    X, y, feats, quality = make_seq('preprocessed_data/combined_data/combined_data.csv',
                           'preprocessed_data/combined_data/ground_truth_combined.csv', return_features=True,
                           trim=(50, 50), step=25)

    X_train, X_test, _, y_test, _, fs_test = train_test_split(X, y, feats, test_size=0.2, random_state=55)

    model = load_model(f'model-normal-fixed.h5')
    t = datetime.datetime.now()
    offsets, _ = evaluate(model, X_test, y_test, fs_test, num_samples=quality, show_ci=False, print_ci=False,
                          label=None)
    print('______\nTime elapsed: ', datetime.datetime.now() - t, '\n')

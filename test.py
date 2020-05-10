import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import pickle

from modelling.feat_engineer.data_processing import make_seq
from tensorflow.keras.models import load_model


def get_conf_int_for(sample_id, X_test, y_test, features=None, num_samples=250, show=True, bins=25):
    X_sample = np.tile(X_test[sample_id], (num_samples, 1, 1))
    if features is not None:
        features = np.tile(features[sample_id], (num_samples, 1))
        ps = model.predict([X_sample, features])
    else:
        ps = model.predict(X_sample)
    if show:
        plt.hist(ps, bins=bins)
        plt.show()
    begin, end = sms.DescrStatsW(ps.flatten()).tconfint_mean()
    print('Confidence interval: ', begin, end)
    print('Mean: ', np.mean([begin, end]))
    print('True: ', y_test[sample_id])


model = load_model('model.h5')
#history = pickle.load(open('/trainHistoryDict'), "rb")

X_test, y_test, fs = make_seq('preprocessed_data/nemcova_data/nemcova_data.csv', 'preprocessed_data/nemcova_data/ground_truths_nemcova.csv', return_features=True)

for i in range(len(y_test[:10])):
    get_conf_int_for(i, X_test, y_test, fs, num_samples=500, show=False)
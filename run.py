import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

from modelling.uncertainty_model.model import create_model
from modelling.feat_engineer.data_processing import *

from keras import optimizers


def get_conf_int_for(sample_id, model, X_test, y_test, features=None, num_samples=250, bins=25):
    X_sample = np.tile(X_test[sample_id], (num_samples, 1, 1))
    if features is not None:
        features = np.tile(features[sample_id], (num_samples, 1))
    ps = model.predict([X_sample, features])
    plt.hist(ps, bins=bins)
    plt.show()
    begin, end = sms.DescrStatsW(ps.flatten()).tconfint_mean()
    print('Confidence interval: ', begin, end)
    print('Mean: ', np.mean([begin, end]))
    print('True: ', y_test[sample_id])


clin_X, clin_y, clin_z = make_seq('preprocessed_data/clinical_data/covital_clinical.csv', 'preprocessed_data/clinical_data/ground_truths_clinical.csv', return_intermediate=True)
com_X, com_y, com_z = make_seq('preprocessed_data/community_data/covital_community.csv', 'preprocessed_data/community_data/ground_truths_community.csv', return_intermediate=True)
sample_X, sample_y, sample_z = make_seq('preprocessed_data/sample_data/sample_data.csv', 'preprocessed_data/sample_data/ground_truths_sample.csv', return_intermediate=True)
nemc_X, nemc_y, nemc_z = make_seq('preprocessed_data/nemcova_data/nemcova_data.csv', 'preprocessed_data/nemcova_data/ground_truths_nemcova.csv', return_intermediate=True)

features = np.hstack([clin_z, com_z, sample_z]).reshape(-1,1)
X = np.vstack([clin_X, com_X, sample_X])
y = np.hstack([clin_y, com_y, sample_y])
data = [X, features]

model = create_model(X, features)

learning_rate = 0.01
opt = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=opt, loss='mse')
# fit model
history_cache = model.fit(data, y, epochs=150, verbose=1, batch_size=29)
model.save('model.h5')

get_conf_int_for(11, model, nemc_X, nemc_y, nemc_z, num_samples=2000, bins=100)

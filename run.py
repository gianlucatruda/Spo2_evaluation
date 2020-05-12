import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import pickle

from modelling.uncertainty_model.model import create_model
from modelling.feat_engineer.data_processing import *

from keras import optimizers


clin_X, clin_y, clin_z = make_seq('preprocessed_data/clinical_data/covital_clinical.csv', 'preprocessed_data/clinical_data/ground_truths_clinical.csv', return_features=True)
com_X, com_y, com_z = make_seq('preprocessed_data/community_data/covital_community.csv', 'preprocessed_data/community_data/ground_truths_community.csv', return_features=True)
sample_X, sample_y, sample_z = make_seq('preprocessed_data/sample_data/sample_data.csv', 'preprocessed_data/sample_data/ground_truths_sample.csv', return_features=True)
nemc_X, nemc_y, nemc_z = make_seq('preprocessed_data/nemcova_data/nemcova_data.csv', 'preprocessed_data/nemcova_data/ground_truths_nemcova.csv', return_features=True)

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

#with open('/trainHistoryDict', 'wb') as f:
#    pickle.dump(history_cache.history, f)

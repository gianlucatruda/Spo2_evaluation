import pickle

from modelling.uncertainty_model.model import create_model, learn, evaluate
from modelling.feat_engineer.data_processing import *
from sklearn.model_selection import train_test_split


X, y, feats = make_seq('preprocessed_data/combined_data/combined_data.csv',
                       'preprocessed_data/combined_data/ground_truth_combined.csv', return_features=True,
                       trim=(50, 50), step=25)

X_train, X_test, y_train, y_test, fs_train, fs_test = train_test_split(X, y, feats, test_size=0.2, random_state=55)

model = create_model(X_train, fs_train)

history_cache = learn(model, X=X_train, y=y_train, features=fs_train, learning_rate=0.01, epochs=10, verbose=1, batch_size=64)
model.save('model.h5')

with open('trainHistoryDict', 'wb') as f:
    pickle.dump(history_cache.history, f)

evaluate(model, X_test, y_test, fs_test, num_samples=500, show_ci=False, print_ci=False)
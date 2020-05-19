import pickle

from modelling.uncertainty_model.model import *
from modelling.feat_engineer.data_processing import *
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    X, y, feats = make_seq('preprocessed_data/combined_data/combined_data.csv',
                           'preprocessed_data/combined_data/ground_truth_combined.csv', return_features=True,
                           trim=(50, 50), step=25)

    X_train, X_test, y_train, y_test, fs_train, fs_test = train_test_split(X, y, feats, test_size=0.2, random_state=55)

    model = create_model(X_train, fs_train)
    learn(model, X=X_train, y=y_train, features=fs_train, learning_rate=0.01, epochs=100, verbose=1, batch_size=64)
    model.save(f'model-normal-fixed.h5')

    evaluate(model, X_test, y_test, fs_test, num_samples=1000, show_ci=False, print_ci=False)

import pickle
import matplotlib.pyplot as plt

from modelling.feat_engineer.data_processing import make_seq
from modelling.uncertainty_model.model import evaluate
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


model = load_model('model.h5')
history = pickle.load(open('trainHistoryDict', "rb"))

X, y, feats = make_seq('preprocessed_data/combined_data/combined_data.csv',
                       'preprocessed_data/combined_data/ground_truth_combined.csv', return_features=True,
                       trim=(50, 50), step=25)

_, X_test, _, y_test, _, fs_test = train_test_split(X, y, feats, test_size=0.2, random_state=55)


e = 50

for dropout in [0.1, 0.2, 0.3, 0.4]:
    print(f'For {e} epochs\n')
    plt.plot(history['loss'])
    plt.show()
    evaluate(model, X_test, y_test, fs_test, num_samples=500, show_ci=False, print_ci=False)
    print('\n\n')
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

from keras.models import Model
from keras.layers import Activation, Concatenate, Dense, Dropout, Input, LSTM
from keras import optimizers


def create_model(X, features):
    '''
    Create a LSTM + static features model
    :param X: sequences from PPG data
    :param features: feature array
    :return: keras model to fit and predict on
    '''
    n_samples = X.shape[0]
    n_steps = X.shape[1]
    n_channels = X.shape[2]
    n_features = features.shape[1]

    seq_inputs = Input(shape=(n_steps, n_channels))
    feat_inputs = Input(shape=(n_features,))

    seq = LSTM(n_samples, activation='tanh', return_sequences=True)(seq_inputs)
    seq = LSTM(n_samples, activation='tanh')(seq)
    seq = Dropout(0.3)(seq, training=True)

    feat = Dense(n_samples, activation='relu')(feat_inputs)

    con = Concatenate()([seq, feat])
    con = Activation('relu')(con)
    con = Dropout(0.3)(con, training=True)

    out = Dense(1, activation='linear')(con)

    return Model(inputs=[seq_inputs, feat_inputs], outputs=out)


def no_feature_model(X):
    n_samples = X.shape[0]
    n_steps = X.shape[1]
    n_channels = X.shape[2]

    inputs = Input(shape=(n_steps, n_channels))
    x = LSTM(n_samples, activation='tanh', return_sequences=True)(inputs)
    x = Dropout(0.2)(x, training=True)
    x = LSTM(n_samples, activation='tanh')(x)
    x = Dense(n_samples, activation='relu')(x)
    x = Dropout(0.2)(x, training=True)
    out = Dense(1, activation='linear')(x)

    return Model(inputs=inputs, outputs=out)


def learn(model, X, y, features, learning_rate=0.01, epochs=100, verbose=1, batch_size=32):
    '''

    :param model: keras model
    :param data: sequences from PPG data
    :param y: target values
    :param learning_rate: (optional)
    :param epochs: (optional)
    :param verbose: (optional)
    :param batch_size: (optional)
    :return: history cache of model fitting
    '''
    opt = optimizers.Adam(lr=learning_rate)
    #opt = optimizers.RMSprop()
    #opt = optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse')
    # fit model
    return model.fit([X, features], y, epochs=epochs, verbose=verbose, batch_size=batch_size)


def get_conf_int_for(model, sample_id, X_test, y_test, features=None, num_samples=250, show_ci=True, print_ci=True,
                     bins=25):
    '''
    Calculation of the 95% confidence interval for model predictions
    :param model: keras model to test
    :param sample_id: id to test
    :param X_test: test sequences from PPG data
    :param y_test: test targets
    :param features: (optional) test features
    :param num_samples: (optional) number of samples for the confidence interval
    :param show_ci: (optional) plot prediction distribution
    :param print_ci: (optional) print confidence interval, mean and true value
    :param bins: (optional) number of bins for prediction distribution
    :return: begin, end: confidence interval
    '''
    X_sample = np.tile(X_test[sample_id], (num_samples, 1, 1))
    if features is not None:
        features = np.tile(features[sample_id], (num_samples, 1))
        ps = model.predict([X_sample, features])
    else:
        ps = model.predict(X_sample)

    begin, end = sms.DescrStatsW(ps.flatten()).tconfint_mean()

    if show_ci:
        plt.hist(ps, bins=bins)
        plt.show()
    if print_ci:
        print('Confidence interval: ', begin, end)
        print('Mean: ', np.mean([begin, end]))
        print('True: ', y_test[sample_id])
        print('Classification: ', (begin <= y_test[sample_id]) and (y_test[sample_id] <= end))
    return begin, end


def evaluate(model, X_test, y_test, fs, num_samples=500, show_ci=False, print_ci=True, bins=25):
    '''
    Model evaluation
    :param model: keras model to evaluate
    :param X_test: test sequences from PPG data
    :param y_test: test targets
    :param fs: test features
    :param num_samples: (optional) number of samples for the confidence interval
    :param show_ci: (optional) plot prediction distribution
    :param print_ci: (optional) print confidence interval, mean and true value
    :param bins: (optional) number of bins for prediction distribution
    '''

    ci_width = []
    dist_from_mean = []
    class_metric = []
    for i in range(len(y_test)):
        begin, end = get_conf_int_for(model, i, X_test, y_test, fs, num_samples=num_samples, show_ci=show_ci,
                                      print_ci=print_ci, bins=bins)
        class_metric.append((begin <= y_test[i]) and (y_test[i] <= end))
        dist_from_mean.append(abs(np.mean([begin, end]) - y_test[i]))
        ci_width.append(end - begin)
    print('Average width: ', np.mean(ci_width))
    plt.hist(dist_from_mean)
    plt.title('Distance from mean')
    plt.show()
    print('Accuracy: ', np.mean(class_metric))
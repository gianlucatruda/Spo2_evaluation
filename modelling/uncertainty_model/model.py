import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import datetime

from keras.models import Model
from keras.layers import Activation, Concatenate, Dense, Dropout, Input, LSTM
from keras import optimizers


def create_no_LSTM_dropout(X, features, dropout=0.4):
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

    feat = Dense(n_samples, activation='relu')(feat_inputs)

    con = Concatenate()([seq, feat])
    con = Activation('relu')(con)
    con = Dropout(dropout)(con, training=True)

    out = Dense(1, activation='linear')(con)

    return Model(inputs=[seq_inputs, feat_inputs], outputs=out)

def create_shallow_LSTM(X, features, dropout=0.5):
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

    seq = LSTM(n_samples, activation='tanh')(seq_inputs)
    seq = Dropout(dropout)(seq, training=True)

    feat = Dense(n_samples, activation='relu')(feat_inputs)

    con = Concatenate()([seq, feat])
    con = Activation('relu')(con)
    con = Dropout(dropout)(con, training=True)

    out = Dense(1, activation='linear')(con)

    return Model(inputs=[seq_inputs, feat_inputs], outputs=out)

def deeper_model(X, features):
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
    seq = Dropout(0.4)(seq, training=True)

    feat = Dense(n_samples, activation='relu')(feat_inputs)

    con = Concatenate()([seq, feat])
    con = Activation('relu')(con)
    con = Dropout(0.4)(con, training=True)

    con = Dense(int(n_samples / 2), activation='relu')(con)
    con = Dropout(0.4)(con, training=True)

    out = Dense(1, activation='linear')(con)

    return Model(inputs=[seq_inputs, feat_inputs], outputs=out)

def even_deeper_model(X, features):
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
    seq = Dropout(0.4)(seq, training=True)

    feat = Dense(n_samples, activation='relu')(feat_inputs)

    con = Concatenate()([seq, feat])
    con = Activation('relu')(con)
    #con = Dropout(0.4)(con, training=True)

    con = Dense(int(n_samples / 2), activation='relu')(con)
    #con = Dropout(0.4)(con, training=True)

    con = Dense(int(n_samples / 4), activation='relu')(con)
    con = Dropout(0.4)(con, training=True)

    out = Dense(1, activation='linear')(con)

    return Model(inputs=[seq_inputs, feat_inputs], outputs=out)


def create_model(X, features, dropout=0.4):
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
    seq = Dropout(dropout)(seq, training=True)

    feat = Dense(n_features, activation='relu')(feat_inputs)

    con = Concatenate()([seq, feat])
    con = Activation('relu')(con)
    con = Dropout(dropout)(con, training=True)

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


def get_conf_int_for(model, sample_id, X_test, y_test, features=None, preds_offset=None, num_samples=250, show_ci=True, print_ci=True,
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
    if type(num_samples) != int:
        distr_size = num_samples[sample_id]
    else:
        distr_size = num_samples

    X_sample = np.tile(X_test[sample_id], (distr_size, 1, 1))
    if features is not None:
        features = np.tile(features[sample_id], (distr_size, 1))
        ps = model.predict([X_sample, features]).flatten()
    else:
        ps = model.predict(X_sample).flatten()

    begin, end = sms.DescrStatsW(ps).tconfint_mean()

    if show_ci:
        plt.hist(ps, bins=bins)
        plt.show()
    if print_ci != 0:
        print('Confidence interval: ', begin, end)
        print('Mean: ', np.mean([begin, end]))
        print('True: ', y_test[sample_id])
        print('Distribution size: ', distr_size)
        print('Classification: ', (begin <= y_test[sample_id]) and (y_test[sample_id] <= end))

    if preds_offset is not None:
        if preds_offset.size == 0:
            return begin, end, ps - y_test[sample_id]
        else:
            return begin, end, np.hstack([preds_offset, ps - y_test[sample_id]])
    else:
        return begin, end


def get_conf_int_for_vectorized(model, X_test, y_test, features=None, num_samples=250, print_ci=True):
    '''
    Calculation of the 95% confidence interval for model predictions
    :param model: keras model to test
    :param sample_id: id to test
    :param X_test: test sequences from PPG data
    :param y_test: test targets
    :param features: (optional) test features
    :param num_samples: (optional) number of samples for the confidence interval
    :param print_ci: (optional) print confidence interval, mean and true value
    :return: begin, end: confidence interval
    '''

    X_sample = np.tile(X_test, (1, num_samples, 1)).reshape(-1, X_test.shape[1], X_test.shape[2])
    fs = np.tile(features, (1, num_samples)).reshape(-1, features.shape[1])
    ps = model.predict([X_sample, fs])

    spl = np.split(ps, X_test.shape[0])

    begins = []
    ends = []
    for i in range(len(spl)):
        begin, end = sms.DescrStatsW(spl[i].flatten()).tconfint_mean()
        begins.append(begin)
        ends.append(end)

    # if print_ci:
    #     print('Confidence interval: ', begin, end)
    #     print('Mean: ', np.mean([begin, end]))
    #     print('True: ', y_test[sample_id])
    #     print('Classification: ', (begin <= y_test[sample_id]) and (y_test[sample_id] <= end))

    return np.array(begins), np.array(ends)


def evaluate(model, X_test, y_test, fs, num_samples=500, show_ci=False, print_ci=True, bins=100, label=None):
    '''
    Model evaluation
    :param model: keras model to evaluate
    :param X_test: test sequences from PPG data
    :param y_test: test targets
    :param fs: test features
    :param num_samples: (optional) number of samples for the confidence interval, or data quality array
    :param show_ci: (optional) plot prediction distribution
    :param print_ci: (optional) print confidence interval, mean and true value
    :param bins: (optional) number of bins for prediction distribution
    '''
    if type(num_samples) != int:
        # rescaling the sizes for the biggest to be 1000
        num_samples = (800 / max(num_samples) * num_samples).astype(int)
    ci_width = []
    dist_from_mean = []
    class_metric = []
    offsets = np.array([])
    for i in range(len(y_test)):
        begin, end, offsets = get_conf_int_for(model, i, X_test, y_test, fs, preds_offset=offsets, num_samples=num_samples,
                                      show_ci=show_ci, print_ci=print_ci, bins=bins)
        class_metric.append((begin <= y_test[i]) and (y_test[i] <= end))
        dist_from_mean.append(abs(np.mean([begin, end]) - y_test[i]))
        ci_width.append(end - begin)
        if print_ci != 0:
            print_ci -= 1
    print('Average width: ', np.mean(ci_width))
    print('Average distance from CI mean to target: ', np.mean(dist_from_mean))
    if label is not None:
        plt.hist(offsets, bins=250)
        plt.title('Average prediction offsets from the target')
        plt.show()
        plt.savefig(f'dist-{label}')
    print('Accuracy: ', np.mean(class_metric))
    return offsets, num_samples
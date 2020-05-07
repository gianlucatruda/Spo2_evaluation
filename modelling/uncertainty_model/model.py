from keras.models import Model
from keras.layers import Activation, Concatenate, Dense, Dropout, Input, LSTM


def create_model(X, features):
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


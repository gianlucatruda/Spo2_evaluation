import numpy as np
import pandas as pd
from modelling.feat_engineer.feature_engineering import rgb_to_ppg, rolling_augment_dataset, \
    engineer_features, select_best_features


def vectorize(full_data, vectorization=np.array, columns=['tx_red', 'tx_green', 'tx_blue']):
    '''
    Vectorization for further sequencing

    :param data: joined ppg data with channels
    :param vectorization: type of vectorization, numpy array or tf.tensor
    :param columns: names of ppg channels, all three RGB by default
    :return:
    '''
    data = full_data.copy()
    data = data.groupby('sample_id').agg(list)
    data['spo2'] = data['spo2'].apply(np.mean)
    data['hr'] = data['hr'].apply(np.mean)

    for column in columns:
        data[column] = data[column].apply(lambda x: vectorization(x, dtype=np.float64))

    return data

def join_data(ppg_data: pd.DataFrame, gt: pd.DataFrame):
    '''
    Joins PPG data and ground truths

    :param ppg_data: data with rgb means and vars
    :param gt: ground truth data
    '''
    gt.columns = gt.columns.str.lower()
    gt = gt.dropna()

    full_data = pd.merge(ppg_data, gt, how='left', left_on='path', right_on='path').drop(columns=['path', 'data_source'])
    return full_data


def prepare_sequences(ppg, gt, channels=['tx_red', 'tx_green', 'tx_blue'], augment=True, trim=(50, 50), step=25, return_full=False):
    '''
    Converts PPG and ground truth data into numpy arrays
    :param ppg: PPG data
    :param gt: Ground truths data
    :param channels: (optional) RGB channels to use in the sequences, all by default
    :param augment: (optional) using augmentation of data
    :param trim: (optional) trimming for rolling augmentation
    :param step: (optional) step for rolling augmentation
    :param return_full: (optional) return joined data as a third parameter
    :return: numpy array of data sequences, targets, and optional full joined data
    '''
    if augment:
        ppg = rolling_augment_dataset(ppg, trim=trim, step=step)
    inter_data = join_data(ppg, gt)
    data = vectorize(inter_data, vectorization=np.array, columns=channels)
    data = data[channels + ['spo2']]  ## spo2 is lowercased in join_data function
    stacked_channels = [np.stack(data[ch].values, axis=0) for ch in channels]
    X = np.stack(stacked_channels, axis=2)
    y = data['spo2'].values
    if return_full:
        return X, y, inter_data
    return X, y


def make_seq(data_path, gt_path, return_features=False, trim=(50, 50), step=25):
    '''
    Create sequences from files with data and ground truths
    :param data_path: path to csv with data
    :param gt_path: path to csv with ground truths
    :param return_features: (optional) return features from joined data as a third parameter
    :param trim: (optional) trimming for rolling augmentation
    :param step: (optional) step for rolling augmentation
    :return: numpy array of data sequences, targets, and optional features
    '''
    data = pd.read_csv(data_path)
    gt = pd.read_csv(gt_path)
    ppg = rgb_to_ppg(data)
    if return_features:
        X, y, z = prepare_sequences(ppg, gt, return_full=return_features, trim=trim, step=step)
        # feature creation
        print(X.shape, y.shape, z.shape)
        print('started feature engineering')
        labels = z[['sample_id', 'spo2']].groupby('sample_id')[['spo2']].agg('mean').reset_index()
        features = engineer_features(z, labels, target='spo2', )
        print('finished feature engineering')
        # new_features = features.loc[:, (features.std() > 100) & (features.mean() < 1000)]
        print('selecting best')
        top50 = select_best_features(features, n_features=50, target='spo2')
        z = top50.drop(columns='spo2').values
        #z = z.groupby('sample_id')['hr'].mean().values
        return X, y, z
    return prepare_sequences(ppg, gt, return_full=return_features, trim=trim, step=step)


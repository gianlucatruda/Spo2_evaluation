import numpy as np
import pandas as pd
from modelling.feat_engineer.feature_engineering import rgb_to_ppg, rolling_augment_dataset


def vectorize(data, vectorization=np.array, columns=['tx_red', 'tx_green', 'tx_blue']):
    '''
    Inplace vectorization for further sequencing
    :param data: joined ppg data with channels
    :param vectorization: type of vectorization, numpy array or tf.tensor
    :param columns: names of ppg channels, all three RGB by default
    :return:
    '''
    for column in columns:
        data[column] = data[column].apply(lambda x: vectorization(x, dtype=np.float64))


def join_data(ppg_data: pd.DataFrame, gt: pd.DataFrame):
    """
    data: data with rgb means and vars
    gt: ground truth data
    """
    gt.columns = gt.columns.str.lower()
    gt = gt.dropna()

    data = pd.merge(ppg_data, gt, how='left', left_on='path', right_on='path').drop(columns=['path', 'data_source'])
    print(data.shape)

    full_data = data.groupby('sample_id').agg(list)
    full_data['spo2'] = full_data['spo2'].apply(np.mean)
    full_data['hr'] = full_data['hr'].apply(np.mean)
    print(full_data.shape)
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
    data = inter_data[channels + ['spo2']]  ## spo2 is lowercased in join_data function
    vectorize(data, vectorization=np.array, columns=channels)
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
        z = z.groupby('sample_id')['hr'].mean().values
        return X, y, z
    return prepare_sequences(ppg, gt, return_full=return_features, trim=trim, step=step)


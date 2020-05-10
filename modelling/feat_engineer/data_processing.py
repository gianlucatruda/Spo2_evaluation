import numpy as np
import pandas as pd
from modelling.feat_engineer.feature_engineering import rgb_to_ppg, rolling_augment_dataset


def vectorize_and_shorten(data, vectorization=np.array, columns=['tx_red', 'tx_green', 'tx_blue'], begin=50, end=290):
    '''
    Function for unaugmented data on a particular interval
    '''
    for column in columns:
        data[column] = data[column].apply(lambda x: vectorization(x[begin:end], dtype=np.float64))


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


def correct_source_paths(df, name='sample_source'):
    '''
    Changing file paths for correct dataframe merging
    :param df:
    :param name:
    :return:
    '''
    data = df.copy()
    data[name] = data[name].apply(lambda x: '/'.join(x.split('/')[:-1]))
    return data


def join_data(ppg_data: pd.DataFrame, gt: pd.DataFrame, unmeasured='Unkown'):
    '''
    Joins PPG data and ground truths (on file paths)
    :param ppg_data: dataframe with rgb means and vars
    :param gt: ground truth data
    :param unmeasured: (optional) label for the absent measurement in ground truth file
    :return: single joined dataframe
    '''

    corr_data = correct_source_paths(ppg_data)
    gt.columns = gt.columns.str.lower()

    gt.loc[gt['spo2'] == unmeasured, 'spo2'] = np.nan
    gt.loc[~gt['spo2'].isna(), 'spo2'] = gt.loc[~gt['spo2'].isna(), 'spo2'].astype(int)
    gt = gt.fillna(gt.mean())
    gt['spo2'] = gt['spo2'].astype(int)  # making SpO2 whole numbers

    # if gt[spo2].dtype == object and gt[spo2].str.contains(unmeasured).sum() > 0:
    #    ## when unmeasured is a string
    #    m = np.mean(gt[gt[spo2] != unmeasured][spo2].astype(int))
    #    gt[spo2] = gt[spo2].replace({unmeasured: m}).astype(int)
    # else:
    #    gt[spo2] = gt[spo2].astype(int)

    data = pd.merge(corr_data, gt, how='left', left_on='sample_source', right_on='path').drop(
        columns=['sample_source', 'path'])

    full_data = data.groupby('sample_id').agg(list)
    full_data['spo2'] = full_data['spo2'].apply(np.mean)
    full_data['hr'] = full_data['hr'].apply(np.mean)
    return full_data


def prepare_sequences(ppg, gt, channels=['tx_red', 'tx_green', 'tx_blue'], augment=True, begin=50, end=290, return_full=False):
    '''
    Converts PPG and ground truth data into numpy arrays
    :param ppg: PPG data
    :param gt: Ground truths data
    :param channels: (optional) RGB channels to use in the sequences, all by default
    :param augment: (optional) using augmentation of data
    :param begin: (optional) start of sequence clipping, useless with augmentation
    :param end: (optional) end of sequence clipping, useless with augmentation
    :param return_full: (optional) return joined data as a third parameter
    :return: numpy array of data sequences, targets, and optional full joined data
    '''
    if augment:
        ppg = rolling_augment_dataset(ppg, trim=(50, 50), step=50)
    inter_data = join_data(ppg, gt)
    data = inter_data[channels + ['spo2']]  ## spo2 is lowercased in join_data function
    if augment:
        vectorize(data, vectorization=np.array, columns=channels)
    else:
        vectorize_and_shorten(data, vectorization=np.array, columns=channels, begin=begin, end=end)
    stacked_channels = [np.stack(data[ch].values, axis=0) for ch in channels]
    X = np.stack(stacked_channels, axis=2)
    y = data['spo2'].values
    if return_full:
        return X, y, inter_data
    return X, y


def make_seq(data_path, gt_path, return_features=False):
    '''
    Create sequences from files with data and ground truths
    :param data_path: path to csv with data
    :param gt_path: path to csv with ground truths
    :param return_features: (optional) return features from joined data as a third parameter
    :return: numpy array of data sequences, targets, and optional features
    '''
    data = pd.read_csv(data_path)
    gt = pd.read_csv(gt_path)
    ppg = rgb_to_ppg(data)
    if return_features:
        X, y, z = prepare_sequences(ppg, gt, return_full=return_features)
        # feature creation
        z = z.groupby('sample_id')['hr'].mean().values
        return X, y, z
    return prepare_sequences(ppg, gt, return_full=return_features)


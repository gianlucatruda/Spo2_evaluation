import numpy as np
import pandas as pd
from modelling.feat_engineer.feature_engineering import rgb_to_ppg


def vectorize_and_shorten(data, vectorization=np.array, columns=['tx_red'], begin=50, end=290):
    for column in columns:
        data[column] = data[column].apply(lambda x: vectorization(x[begin:end], dtype=np.float64))


def prepare_sequences(ppg, gt, channels=['tx_red', 'tx_green', 'tx_blue'], begin=50, end=290, return_intermediate=False):
    inter_data = join_data(ppg, gt)
    data = inter_data[channels + ['spo2']]  ## spo2 is being lowercased in join_data function
    vectorize_and_shorten(data, vectorization=np.array, columns=channels, begin=begin, end=end)
    stacked_channels = [np.stack(data[ch].values, axis=0) for ch in channels]
    X = np.stack(stacked_channels, axis=2)
    y = data['spo2'].values
    if return_intermediate:
        return X, y, inter_data
    return X, y


def make_seq(data_path, gt_path, return_intermediate=False):
    data = pd.read_csv(data_path)
    gt = pd.read_csv(gt_path)
    ppg = rgb_to_ppg(data)
    if return_intermediate:
        X, y, z = prepare_sequences(ppg, gt, return_intermediate=return_intermediate)
        z = z.groupby('sample_id')['hr'].mean().values
        return X, y, z
    return prepare_sequences(ppg, gt, return_intermediate=return_intermediate)


def correct_source_paths(tmp, name='sample_source'):
    data = tmp.copy()
    data[name] = data[name].apply(lambda x: '/'.join(x.split('/')[:-1]))
    return data


def join_data(ppg_data: pd.DataFrame, gt: pd.DataFrame, unmeasured='Unkown'):
    """
    data: data with rgb means and vars
    gt: ground truth data
    unmeasured: (optional) label for the absent measurement in ground truth file
    spo2: (optional) label for SpO2 measurements
    hr: (optional) label for HR measurements
    """
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


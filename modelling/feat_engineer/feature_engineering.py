from sklearn.exceptions import DataConversionWarning
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, firwin
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
import pywt
from ..lamonaca_and_nemcova.utils import spo2_estimation, get_peak_height_slope
from tqdm import tqdm


def _eval_feat_engineering(all_features):
    # Little helper function I wrote for some analysis comparing HR and SpO2 as a target

    # https://github.com/gianlucatruda/Spo2_evaluation/issues/4
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.linear_model import LassoLars, LinearRegression
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from tqdm import tqdm
    from sklearn.model_selection import KFold
    import json
    from pandas.io.json import json_normalize

    import warnings
    warnings.filterwarnings(action='ignore', category=UserWarning)
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)

    targets = ['spo2', 'hr']
    models = [XGBRegressor, LassoLars, LinearRegression]
    model_names = ['xgboost', 'lasso', 'lr']
    K_vals = [2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]

    results = {'target': [], 'model': [], 'k': [], 'mse': []}

    for target in targets:
        print(f"Evaluating {target}...")
        for K in tqdm(K_vals, total=2*len(K_vals)):
            _X = all_features.drop(targets, axis=1).values
            _y = all_features[target].values
            kbest = SelectKBest(f_classif, k=K).fit(_X, _y)
            _X = _X[:, kbest.get_support()]
            kf = KFold(n_splits=10)
            for train_index, test_index in kf.split(_X):
                X_train, X_test = _X[train_index], _X[test_index]
                y_train, y_test = _y[train_index], _y[test_index]
                for model, mname in zip(models, model_names):
                    mod = model().fit(X_train, y_train)
                    score = mse(y_test, mod.predict(X_test))
                    for key, value in {'target': target, 'model': mname, 'mse': score, 'k': K}.items():
                        results[key].append(value)

    return pd.DataFrame.from_dict(results)


def calc_spo2(df: pd.DataFrame):
    """Not working"""
    raise NotImplementedError()

    e_hb_600 = 15.0
    e_hb_940 = 0.8
    e_hbo_600 = 3.5
    e_hbo_940 = 1.1

    timestamps = np.linspace(0, int(float(df.shape[0])/30.0), num=df.shape[0])

    vp_940, m_940, hr_940 = get_peak_height_slope(
        df['tx_green'].values, timestamps, 30.0)
    vp_600, m_600, hr_600 = get_peak_height_slope(
        df['tx_red'].values, timestamps, 30.0)

    ln_vp_600 = np.log(vp_600)
    ln_vp_940 = np.log(vp_940)

    print(ln_vp_600, ln_vp_940)
    print(m_600, m_940)
    print(hr_600, hr_940)

    spo2 = (e_hb_600 * np.sqrt(m_940 * ln_vp_940) - e_hb_940 * np.sqrt(m_600 * ln_vp_600)) / \
        (np.sqrt(m_940 * ln_vp_940)*(e_hb_600 - e_hbo_600) -
         np.sqrt(m_600 * ln_vp_600)*(e_hb_940 - e_hbo_940))

    return spo2.mean(), spo2.std()


def _wavelet_filter_signal(s, wave='db4', *args, **kwargs):
    cA, cD = pywt.dwt(s, wave)
    cD_mod = pywt.threshold(cD, *args, **kwargs)
    s_mod = pywt.idwt(cA, cD_mod, wave)
    return s_mod


def rgb_to_ppg(df: pd.DataFrame,
               filter='band',
               outlier_thresh=2.5,
               scale=True,
               smoothing_window=3,
               block_id='sample_id',
               ) -> pd.DataFrame:
    """Turn RGB means into PPG curves as per Lamonaca.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the colour data per frame (`mean_red`, etc.)
    filter : str, optional
        The kind of signal filter to apply, by default 'band'.
        `band` = bandpass Butterworth filter
        `low` = lowpass Butterworth filter
        `firwin` = lowpass Firwin filter
    outlier_thresh : float, optional
        The number of standard deviations at which outliers will be
        clipped, by default 2.5.
    scale : bool, optional
        Whether the signals should be scaled to the range 0-1 using MinMax.
        Set to None to disable.
    smoothing_window : int, optional
        The window size to use for smoothing. Set to None to disable.
    block_id : str, optional
        The field to use for chunking the data before applying transforms,
        by default 'sample_id'.

    Returns
    -------
    pd.DataFrame
        Copy of the original dataframe, but with 'tx_red', 'tx_blue', 'tx_green'.

    """

    if filter not in ['band', 'low', 'firwin', None]:
        raise ValueError("filter must be 'band', 'low', 'firwin', or None")

    _df = df.copy()

    for i, colour in enumerate(['red', 'green', 'blue']):
        _df[f'tx_{colour}'] = _df[f'mean_{colour}'] * 1.0

    tx_fields = ['tx_red', 'tx_green', 'tx_blue']

    '''Filtering'''
    fs = 30.0       # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2

    if filter is not None:
        if filter == 'band':
            low = 0.3 / nyq
            high = 4.2 / nyq
            b, a = butter(order, [low, high], btype='band', analog=False)
        else:
            cutoff = 4.0 / nyq    # desired cutoff frequency of the filter, Hz
            if filter == 'low':
                b, a = butter(order, cutoff, btype='low', analog=False)
            if filter == 'firwin':
                b = firwin(order+1, cutoff, )
                a = 1.0

    for bid in tqdm(_df[block_id].unique()):
        for field in tx_fields:

            # Apply the filter
            if filter is not None:
                _df.loc[_df[block_id] == bid, field] = filtfilt(
                    b, a, _df[_df[block_id] == bid][field])

            # Clip outliers
            if outlier_thresh is not None:
                sig = _df[_df[block_id] == bid][field]
                stdev = sig.values.std()
                median = np.median(sig.values)
                mean = sig.mean()
                sig = np.where(sig < (outlier_thresh*stdev),
                               sig, mean+outlier_thresh*stdev)
                sig = np.where(sig < (-outlier_thresh*stdev),
                               mean-outlier_thresh*stdev, sig)
                _df.loc[_df[block_id] == bid, field] = sig

            # Min-Max scaling
            if scale:
                _df.loc[_df[block_id] == bid, field] = MinMaxScaler(
                ).fit_transform(_df[_df[block_id] == bid][field].values.reshape(-1, 1))

            # Apply smoothing
            if smoothing_window is not None:
                smoothed = _df.loc[_df[block_id] ==
                                   bid, field].rolling(2).mean()
                _df.loc[_df[block_id] == bid, field] = smoothed

    return _df


def _create_path_field(df: pd.DataFrame) -> pd.DataFrame:
    """Create `path` field from `sample_source`, then remove `sample_source`.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries dataset from Nemcova or sample data.

    Returns
    -------
    pd.DataFrame
        A copy of the input data, but with `path` field instead of `sample_source`.
    """
    _df = df.copy()
    _df['path'] = _df['sample_source'].apply(lambda x: str(Path(x).parent))
    _df.drop('sample_source', axis=1, inplace=True)

    return _df


def _attach_sample_id_to_ground_truth(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of `labels` with added `sample_id` field.

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    labels : pd.DataFrame
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """

    _labels = labels.copy()
    _df = df.drop_duplicates(subset=['sample_id'], keep='first')

    if 'path' not in _df.columns:
        _df = _create_path_field(_df)

    out = _df.merge(_labels, on='path', how='inner')

    return out


def engineer_features(df: pd.DataFrame, labels: pd.DataFrame, target='SpO2', select=True):
    """Automatic feature engineering (with optional selection) for timeseries dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of rgb and/or ppg timeseries for all subjects.
    labels : pd.DataFrame
        A dataframe matching sample IDs to ground truth (e.g. SpO2)
    target : str, optional
        The name of the column you're trying to predict, by default 'SpO2'
    select : bool, optional
        Whether to automatically filter down to statistically significant features, by default True

    Returns
    -------
    pd.DataFrame
        A dataframe with new features added, including the target feature.
    """

    _df = df.copy()
    _labels = labels.copy()

    # if 'sample_id' not in _labels.columns:
    #     _labels = _attach_sample_id_to_ground_truth(_df, _labels)
    _df = _df.select_dtypes(np.number)

    ids = _df['sample_id'].unique()

    _labels = _labels[_labels['sample_id'].isin(ids)]
    y = _labels.set_index('sample_id')[target].astype(np.float)

    if 'sample_source' in _df.columns:
        _df.drop('sample_source', axis=1, inplace=True)

    extracted_features = extract_features(
        _df,
        column_id="sample_id",
        column_sort="frame",
    )

    impute(extracted_features)
    features = extracted_features

    if select:
        features_filtered = select_features(
            extracted_features, y, ml_task='regression',)
        print(extracted_features.shape, features_filtered.shape)
        features = features_filtered

    out_df = features.join(y, how='left')

    return out_df


def select_best_features(df: pd.DataFrame, n_features=20, target='SpO2') -> pd.DataFrame:
    """Perform feature selection using k-best strategy and return dataframe
    including the target.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with all features present.
    n_features : int, optional
        k in k-best algorithm (the number of features), by default 20
    target : str, optional
        the target feature name, by default 'SpO2'

    Returns
    -------
    pd.DataFrame
        A copy of the original dataframe, with only the k-best features.
    """

    kbest = SelectKBest(f_classif, k=n_features)
    _X = df.drop(target, axis=1).values
    _y = df[target].values
    kbest.fit(_X, _y)

    indices = kbest.get_support()
    best_cols = df.drop(target, axis=1).columns[indices]

    _df = df[[*best_cols, target]]

    return _df


def rolling_augment_dataset(df: pd.DataFrame, n_frames=200, trim=(20, 20), step=10):
    """Rolling window of `n_frames` to augment the data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw RGB signal data with `sample_id` and `frame` fields.
    n_frames : int, optional
        The number of frames per window (the width), by default 200
    trim : tuple, optional
        How many frames to trim of the beginning and end of each original
        sample, by default (20, 20)
    step : int, optional
        The number of frames to shift the window start for each
        augmented sample, by default 10

    Returns
    -------
    out_df : pd.DataFrame
        A dataframe of the augmented data that should be much longer than
        the previous dataframe. The `sample_id` and `frame` fields are
        updated from the original values passed in.
    """

    augmented_blocks = []
    for sid in tqdm(df.sample_id.unique()):
        block = df[df.sample_id == sid].iloc[trim[0]: -trim[1]].copy()
        for win_index, start in enumerate(range(0, block.shape[0] - n_frames, step)):
            sid_new = sid*1000 + win_index
            window = block.iloc[start: start + n_frames].copy()
            window['sample_id'] = sid_new
            window['frame'] = np.arange(0, n_frames)
            augmented_blocks.append(window)
    out_df = pd.concat(augmented_blocks)

    return out_df


def augment_dataset(df: pd.DataFrame, n_frames=200, numeric_sample_id=True) -> pd.DataFrame:
    """Simple data augmentation by chopping timeseries into blocks.

    This doesn't modify the data. It just overwrites the `sample_id`
    field with new values. i.e. a single sample of `N` frames is now
    seen as `M` different samples of `N/M` frames.

    Parameters
    ----------
    df : pd.DataFrame
        Your timeries data for all samples.
    n_frames : int, optional
        The lenth (in frames) of each new chunk/sample, by default 200

    Returns
    -------
    pd.DataFrame
        The augmented timeseries data.show()
    """

    def gen_sample_id(sid, frame):
        if numeric_sample_id:
            return int(100*sid + (frame // n_frames))
        else:
            return f"{sid}_{frame // n_frames}"

    def gen_frame(frame):
        return frame % n_frames

    _df = df.copy()

    # Make a copy of the sample_id for later reference
    _df['original_sample_id'] = _df['sample_id']

    # Apply the gen_sample_id function to create an augmented sample_id
    _df['sample_id'] = _df.apply(lambda x: gen_sample_id(
        x['original_sample_id'], x['frame']), axis=1)

    # Apply the gen_frame function to augment the frame values
    _df['frame'] = _df['frame'].apply(gen_frame)

    # Trim the excess (samples which have fewer than `n_frames` frames)
    drop_sids = []
    for sid in _df['sample_id'].unique():
        frame_count = _df[_df['sample_id'] == sid]['frame'].tail(1).values
        if (frame_count + 1) < n_frames:
            drop_sids.append(sid)
    _df = _df[~_df['sample_id'].isin(drop_sids)]

    return _df

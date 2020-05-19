import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, firwin
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import scipy
from scipy.special import kl_div
from modelling.feat_engineer.feature_engineering import _attach_sample_id_to_ground_truth
from tqdm import tqdm


def estimate_quality(data: pd.DataFrame, labels: pd.DataFrame, prefix='tx_') -> pd.DataFrame:
    # Estimate signal quality with a composite metric
    _labels = labels.copy()

    if 'sample_id' not in _labels.columns:
        _labels = _attach_sample_id_to_ground_truth(data, _labels)
        print('ping')

    _labels['quality'] = np.nan

    for sid in tqdm(data['sample_id'].unique()):
        block = data[data['sample_id'] == sid]

        kld = kl_divergence(block, prefix=prefix)
        skews = []
        for c in ['red', 'green', 'blue']:
            skews.append(scipy.stats.skew(block[f"{prefix}{c}"]))

        # Calculate score
        abs_skews = [np.abs(s) for s in skews]
        sum_abs_skews = np.sum(abs_skews)
        scaled_kld = min(kld / block.shape[0], 1.0)
        scaled_skew = min(sum_abs_skews / block.shape[0], 1.0)
        print(scaled_skew, scaled_kld)
        _labels.loc[_labels['sample_id'] == sid, 'quality'] = scaled_skew + scaled_kld


    return _labels


def kl_divergence(df: pd.DataFrame, prefix='tx_', band=(0.5, 3.5)):
    """Generate a signal quality score (lower is better) based on KL divergence.

    Parameters
    ----------
    df : pd.DataFrame
        input data that (works better if already cleaned)
    prefix : str, optional
        the prefix to look for with each colour, by default 'tx_'
    band : tuple, optional
        The bounds of the frequency band within which to measure KL divergence, by default (0.5, 3.5)

    Returns
    -------
    total : float
        Sum of KL divergence
    """
    colours = ['red', 'green', 'blue']

    def get_spec(block, field):
        x = block[field]
        f, Pxx_spec = welch(x, 30.0, 'flattop',
                            nperseg=len(x), scaling='spectrum')
        # only consider certain Hz band
        spec = Pxx_spec[(f < band[1]) & (f > band[0])]
        return spec

    specs = {c: get_spec(df, f"{prefix}{c}") for c in colours}
    klds = 0
    for colour1 in colours:
        for colour2 in colours:
            if colour1 != colour2:
                # Calculate KL-divergence between pairs of colours
                kld = kl_div(specs[colour1], specs[colour2])
                klds += np.sum(kld)
    total = np.sum(klds)
    return total


def skewness(df: pd.DataFrame, prefix='mean_', id_field='sample_id') -> pd.DataFrame:
    """Returns flattened form of the dataframe, with skewness values for each colour channel

    Parameters
    ----------
    df : pd.DataFrame
        Input data (time series)
    prefix : str, optional
        The prefix to look for with each colour, by default 'mean_'
    id_field : str, optional
        The field to chunk the data by, by default 'sample_id'

    Returns
    -------
    pd.DataFrame
        Dataframe with the skewness values.
    """

    skews = {'sample_id': [], 'skew_red': [],
             'skew_green': [], 'skew_blue': []}

    for sid in df[id_field].unique():
        subset = df[df[id_field] == sid]
        for c in ['red', 'green', 'blue']:
            skews[f"skew_{c}"].append(scipy.stats.skew(subset[f"{prefix}{c}"]))
        skews['sample_id'].append(sid)

    return pd.DataFrame(skews)


def zero_crossings(data: np.array) -> int:
    """ The rate of sign-changes in the processed signal

    Parameters
    ----------
    data : np.array
        Your array of signal data

    Returns
    -------
    int
        The number of times the signal crossed the line y = 0.

    Implementation: https://stackoverflow.com/questions/30272538/python-code-for-counting-number-of-zero-crossings-in-an-array#30281079
    """

    return ((data[:-1] * data[1:]) < 0).sum()


def power_analysis(df: pd.DataFrame, field_prefix='mean_', show=False):
    """Generate Welch Power Spectrum plot on RGB channels.

    NOTE: Heavily adapted from https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python


    Parameters
    ----------
    df : pd.DataFrame
        Sample data with RGB channels.
    field_prefix : str, optional
        The prefix for fields to consider when, by default 'mean_'
    show : bool, optional
        Whether to call plt.show() automatically, by default False
    """

    fs = 30.0       # sample rate, Hz

    plt.figure()
    for colour in ['red', 'green', 'blue']:
        x = df[f"{field_prefix}{colour}"]
        f, Pxx_spec = welch(x, fs, 'flattop', 1024, scaling='spectrum')
        plt.semilogy(f, np.sqrt(Pxx_spec), color=colour)

    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.title('Power spectra (welch)')

    if show:
        plt.show()


def inspect_ppg(df: pd.DataFrame, prefix='tx_', xlim=None, title=None, *args, **kwargs):
    """Renders subplots of PPG data for each colour.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'tx_red', 'tx_green', 'tx_blue' fields.
    xlim : Tuple
        the range of frames to show, optional, by default None

    Returns
    -------
    Figure
    """

    fig, ax = plt.subplots(3, 1)

    for i, colour in enumerate(['red', 'green', 'blue']):
        field = f"{prefix}{colour}"
        ax[i].plot(df['frame'], df[field], color=colour, *args, **kwargs)
        if xlim is not None:
            ax[i].set_xlim(xlim)
    if title is not None:
        fig.suptitle(title)
    ax[1].set_ylabel('Value')
    ax[2].set_xlabel('Frame')

    return fig

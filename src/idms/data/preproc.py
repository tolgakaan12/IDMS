import numpy as np
from scipy import signal
import scipy as sp
# from EMG_filter.lms_filt import anc
# from EMG_filter.lms_filt import spectrum_lms
import warnings


def smooth(data):
    return np.apply_along_axis(lambda d: signal.medfilt(d, kernel_size=21), axis=1, arr=data)


def angle_shift(ang):
    if np.mean(ang) > 2.8:
        return ang - np.pi
    if np.mean(ang) < -2.8:
        return ang + np.pi
    return ang


def unwrap_shift(data):
    return np.apply_along_axis(angle_shift, axis=1, arr=data)


# def lms_anc(data, clear_ecg=False, filter_mask=None, **kwargs):
#     data = norm_emg(data)
#     filter_mask = np.arange(len(data)) if filter_mask is None else filter_mask
#     for ch in filter_mask:
#         s = data[ch]
#         ref = data[-1]
#         F = anc(s=s, ref=ref, **kwargs)
#         data[ch] = (F.e)
#     if clear_ecg:
#         data = data[1:-1]
#     return data


def norm_emg(data, **kwargs):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]


def nan_tolerant_norm_emg(data, **kwargs):
    non_nan = ~np.any(np.isnan(data), axis=0)
    emg_std = np.std(data[:, non_nan], axis=1)
    emg_mean = np.mean(data[:, non_nan], axis=1)
    out = np.empty_like(data)
    out[:] = np.nan
    out[:, non_nan] = (data[:, non_nan] - emg_mean[:, None]) / emg_std[:, None]
    return out


def bp_filter(data, high_band=7, low_band=400, sfreq=2000, filt_ord=4, causal=True, **kwargs):
    data = norm_emg(data)
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)
    # create bandpass filter for EMG
    b, a = signal.butter(filt_ord, [high_band, low_band], btype='bandpass', output='ba')
    # process EMG signal: filter EMG
    return signal.lfilter(b, a, data, axis=1) if causal else signal.filtfilt(b, a, data, axis=1)


def denoise(emg, sfreq=2000, high_band=20, low_band=450):
    """
    Proper EMG denoising with bandpass filtering and notch filtering.
    
    Args:
        emg: EMG data array (single channel or channels x samples)
        sfreq: Sampling frequency (Hz)
        high_band: High-pass cutoff frequency (Hz)
        low_band: Low-pass cutoff frequency (Hz)
        
    Returns:
        Filtered EMG data
    """
    # Handle both single channel and multi-channel inputs
    if emg.ndim == 1:
        # Single channel
        emg_processed = emg - np.mean(emg)
        # normalise cut-off frequencies to sampling frequency
        high_band_norm = high_band / (sfreq / 2)
        low_band_norm = low_band / (sfreq / 2)

        # create bandpass filter for EMG
        ba = sp.signal.butter(4, [high_band_norm, low_band_norm], btype='bandpass', output='ba')
        b, a = sp.signal.iirnotch(50, fs=sfreq, Q=30)
        emg_notched = sp.signal.filtfilt(b, a, emg_processed)
        # process EMG signal: filter EMG
        emg_filtered = sp.signal.filtfilt(ba[0], ba[1], emg_notched)
        return emg_filtered
    else:
        # Multi-channel - apply to each channel
        emg_processed = np.zeros_like(emg)
        for i in range(emg.shape[0]):
            channel_emg = emg[i, :] - np.mean(emg[i, :])
            # normalise cut-off frequencies to sampling frequency
            high_band_norm = high_band / (sfreq / 2)
            low_band_norm = low_band / (sfreq / 2)

            # create bandpass filter for EMG
            ba = sp.signal.butter(4, [high_band_norm, low_band_norm], btype='bandpass', output='ba')
            b, a = sp.signal.iirnotch(50, fs=sfreq, Q=30)
            emg_notched = sp.signal.filtfilt(b, a, channel_emg)
            # process EMG signal: filter EMG
            emg_processed[i, :] = sp.signal.filtfilt(ba[0], ba[1], emg_notched)
        return emg_processed


def hp_filter_120hz(data, cutoff=120, sfreq=2000, filt_ord=4, causal=True, **kwargs):
    """
    4th order 120Hz high-pass filter for EMG data.
    Based on study methodology for EMG preprocessing.
    
    Args:
        data: EMG data array (channels, samples)
        cutoff: High-pass cutoff frequency (Hz)
        sfreq: Sampling frequency (Hz)
        filt_ord: Filter order
        causal: If True, use causal filter (lfilter), else non-causal (filtfilt)
        
    Returns:
        Filtered EMG data
    """
    data = norm_emg(data)
    # normalise cut-off frequency to sampling frequency
    nyquist = sfreq / 2
    high_norm = cutoff / nyquist
    
    # create high-pass filter
    b, a = signal.butter(filt_ord, high_norm, btype='highpass', output='ba')
    
    # process EMG signal: filter EMG
    return signal.lfilter(b, a, data, axis=1) if causal else signal.filtfilt(b, a, data, axis=1)


# def spec_proc(data, fbins=500, gamma=0.01, frange=(0, 500), fsamp=2000):
#     data = norm_emg(data)
#     return np.array([np.abs(
#         spectrum_lms(emg, fbins, gamma=gamma).W[int(fbins*frange[0]/fsamp):int(fbins*frange[1]/fsamp), 1:]).T
#                      for emg in data])


def mav(data, ma_window_size=201, method='ma', **kwargs):
    warnings.warn('MAV filter introduces delay. Be sure to add that delay to generator.')

    if data.ndim != 2 or data.shape[1] < data.shape[0]:
        raise ValueError("MAV implementation configured for 2D Arrays, where second axis is time")

    def moving_average(a, n=3):
        a = np.concatenate((a[:, :n-1], a), axis=1)
        ret = np.cumsum(a, axis=1, dtype=float)
        ret[:, n:] = ret[:, n:] - ret[:, :-n]
        return ret[:, n - 1:] / n

    def hamm_smooth(a, n=3):
        a = np.concatenate((a[:, :n - 1], a), axis=1)
        ret = np.apply_along_axis(lambda m: np.convolve(m, np.hamming(n), 'valid')/n, axis=1, arr=a)
        return ret

    procced = norm_emg(data)
    procced = np.abs(procced)
    if method == 'hamm':
        return hamm_smooth(procced, ma_window_size)
    return moving_average(procced, ma_window_size)


# def iterative_demean(data, **kwargs):
#     import EMG_filter.lms as lms
#     x_in = np.empty((0, data.shape[0]))
#     d_in = data
#     # mean_estimator = lms.GNGD(x_in=x_in, d_in=d_in, mu=0.8, bias=1, beta=0.0005)
#     mean_estimator = lms.LMS(x_in=x_in, d_in=d_in, bias=1, **kwargs)
#     mean_estimator.pretrain(2000, 100)
#     mean_estimator.run()
#     d_in = d_in - mean_estimator.Y
#     return d_in


def empty_channels(data):
    return np.empty((0, data.shape[1]))

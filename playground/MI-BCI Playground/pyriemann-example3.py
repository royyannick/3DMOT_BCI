#https://pyriemann.readthedocs.io/en/latest/auto_examples/SSVEP/plot_classify_ssvep_mdm.html#sphx-glr-auto-examples-ssvep-plot-classify-ssvep-mdm-py

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

from mne import get_config, set_config, find_events, create_info, Epochs
from mne.datasets.utils import _get_path
#from mne.utils import _fetch_file, _url_to_local_path
from mne.io import Raw, RawArray

from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM

from sklearn.model_selection import cross_val_score, RepeatedKFold

data_path = sample.data_path()

# %%
def download_sample_data(dataset="ssvep", subject=1, session=1):
    """Download BCI data for example purpose

    Parameters
    ----------
    dataset : str
        type of the dataset, could be "ssvep", "p300" or "imagery"
        Default is "ssvep", as other are not implemented
    subject : int
        Subject id, dataset specific (default: 1)
    session : int, default 1
        Session number%load , dataset specific (default: 1)

    Returns
    -------
    destination : str
        Path to downloaded data
    """
    if dataset == "ssvep":
        DATASET_URL = 'https://zenodo.org/record/2392979/files/'
        url = '{:s}subject{:02d}_run{:d}_raw.fif'.format(DATASET_URL,
                                                         subject, session + 1)
        sign = 'SSVEPEXO'
        key, key_dest = 'MNE_DATASETS_SSVEPEXO_PATH', 'MNE-ssvepexo-data'
    elif dataset == "p300" or dataset == "imagery":
        raise NotImplementedError("Not yet implemented")

    # Use MNE _fetch_file to download EEG file
    if get_config(key) is None:
        set_config(key, os.path.join(os.path.expanduser("~"), "mne_data"))
    path = _get_path(None, key, sign)
    destination = _url_to_local_path(url, os.path.join(path, key_dest))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        _fetch_file(url, destination, print_destination=False)
    return destination


# Download data
destination = download_sample_data(dataset="ssvep", subject=12, session=1)
# Read data in MNE Raw and numpy format
raw = Raw(destination, preload=True, verbose='ERROR')
events = find_events(raw, shortest_event=0, verbose=False)
raw = raw.pick_types(eeg=True)
event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
sfreq = int(raw.info['sfreq'])

eeg_data = raw.get_data()

# %%
n_seconds = 2
time = np.linspace(0, n_seconds, n_seconds * sfreq, endpoint=False)[np.newaxis, :]
plt.figure(figsize=(10, 4))
plt.plot(time.T, eeg_data[np.array(raw.ch_names) == 'Oz', :n_seconds*sfreq].T,
         color='C0', lw=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"Oz ($\mu$V)")
plt.show()

# %%
plt.figure(figsize=(10, 4))
for ch_idx, ch_name in enumerate(raw.ch_names):
    plt.plot(time.T, eeg_data[ch_idx, :n_seconds*sfreq].T, lw=0.5,
             label=ch_name)
plt.xlabel("Time (s)")
plt.ylabel(r"EEG ($\mu$V)")
plt.legend(loc='upper right')
plt.show()

# %%
raw.plot(duration=n_seconds, start=0, n_channels=8, scalings={'eeg': 4e-2},
         color={'eeg': 'steelblue'})
# %%
def _bandpass_filter(signal, lowcut, highcut):
    """ Bandpass filter using MNE """
    return signal.copy().filter(l_freq=lowcut, h_freq=highcut,
                                method="iir").get_data()


# We stack the filtered signals to build an extended signal
frequencies = [13., 17., 21.]
freq_band = 0.1
ext_signal = np.vstack([_bandpass_filter(raw,
                                         lowcut=f-freq_band,
                                         highcut=f+freq_band)
                        for f in frequencies])

# %%
info = create_info(
    ch_names=sum(list(map(lambda s: [ch+s for ch in raw.ch_names],
                          ["-13Hz", "-17Hz", "-21Hz"])), []),
    ch_types=['eeg'] * 24,
    sfreq=sfreq)

raw_ext = RawArray(ext_signal, info)
raw_ext.plot(duration=n_seconds, start=14, n_channels=24,
             scalings={'eeg': 5e-4}, color={'eeg': 'steelblue'})

# %%
epochs = Epochs(raw_ext, events, event_id, tmin=2, tmax=5, baseline=None)

n_seconds = 3
time = np.linspace(0, n_seconds, n_seconds * sfreq,
                   endpoint=False)[np.newaxis, :]
channels = range(0, len(raw_ext.ch_names), len(raw.ch_names))
plt.figure(figsize=(7, 5))
for f, c in zip(frequencies, channels):
    plt.plot(epochs.get_data()[5, c, :].T, label=str(int(f))+' Hz')
plt.xlabel("Time (s)")
plt.ylabel(r"Oz after filtering ($\mu$V)")
plt.legend(loc='upper right')
plt.show()

# %%
cov_ext_trials = Covariances(estimator='lwf').transform(epochs.get_data())

# This plot shows an example of a covariance matrix observed for each class:

plt.figure(figsize=(7, 7))
for i, l in enumerate(event_id):
    ax = plt.subplot(2, 2, i+1)
    plt.imshow(cov_ext_trials[events[:, 2] == event_id[l]][0],
               cmap=plt.get_cmap('RdBu_r'))
    plt.title('Cov for class: '+l)
    plt.xticks([])
    if i == 0 or i == 2:
        plt.yticks(np.arange(len(info['ch_names'])), info['ch_names'])
        ax.tick_params(axis='both', which='major', labelsize=7)
    else:
        plt.yticks([])
plt.show()

# %%
cov_centers = np.empty((len(event_id), 24, 24))
for i, l in enumerate(event_id):
    cov_centers[i] = mean_riemann(cov_ext_trials[events[:, 2] == event_id[l]])

plt.figure(figsize=(7, 7))
for i, l in enumerate(event_id):
    ax = plt.subplot(2, 2, i+1)
    plt.imshow(cov_centers[i], cmap=plt.get_cmap('RdBu_r'))
    plt.title('Cov mean for class: '+l)
    plt.xticks([])
    if i == 0 or i == 2:
        plt.yticks(np.arange(len(info['ch_names'])), info['ch_names'])
        ax.tick_params(axis='both', which='major', labelsize=7)
    else:
        plt.yticks([])
plt.show()
# %%

# %%

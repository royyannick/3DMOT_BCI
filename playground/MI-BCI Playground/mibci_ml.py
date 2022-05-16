import mne
import os
from mne.channels.montage import make_standard_montage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

%matplotlib qt

#%%
subject = 1
runs = [6, 10, 14]
raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload = True) for f in raw_fnames])

#%%
eegbci.standardize(raw)
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)

#%%
raw.filter(7., 30.)
events, _ = events_from_annotations(raw, event_id=dict(T1=0, T2=1))
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

#%%
tmin, tmax = -1., 4.
epochs = Epochs(raw, events, None, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1]

#%%
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

#%%
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# %%
csp.fit_transform(epochs_data, labels)
csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

# %%
sfreq = raw.info['sfreq']
w_length = int(sfreq * 0.5)
w_step = int(sfreq * 0.1)
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
scores_windows = []

#%%
for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    lda.fit(X_train, y_train)

    score_this_window =[]
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:,:, n:(n+w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

#%%
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Change')
plt.xlabel('time(s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()


# Doesn't seem to work. Not sure I understand the Generalizing Estimator.
#%%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import GeneralizingEstimator

# %%
epochs_data_train = epochs_train.get_data()

clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1,
                                 verbose=True)

# Fit classifiers on the epochs where the stimulus was presented to the left.
# Note that the experimental condition y indicates auditory or visual
time_gen.fit(X=epochs_train[0:40].get_data(),
             y=labels[0:40])

# %%
scores = time_gen.score(X=epochs_train[-5:].get_data(),
                        y=labels[-5:])

#%%
fig, ax = plt.subplots(1)
im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r', origin='lower',
                extent=epochs.times[[0, -1, 0, -1]])
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Generalization across time and condition')
plt.colorbar(im, ax=ax)
plt.show()
# %%

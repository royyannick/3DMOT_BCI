#https://pyriemann.readthedocs.io/en/latest/auto_examples/ERP/plot_classify_EEG_tangentspace.html#sphx-glr-auto-examples-erp-plot-classify-eeg-tangentspace-py

#%%
import numpy as np

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix

import mne
from mne import io
from mne.datasets import sample
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

data_path = sample.data_path()

#%%
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks, baseline=None, preload=True, verbose=False)

labels = epochs.events[:, -1]
evoked = epochs.average()

# %%
n_components = 2

cv = KFold(n_splits=10, shuffle=True, random_state=42)
epochs_data = epochs.get_data()

clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds = np.zeros(len(labels))

for train_idx, test_idx in cv.split(epochs_data):
    y_train, y_test = labels[train_idx], labels[test_idx]

    clf.fit(epochs_data[train_idx], y_train)
    preds[test_idx] = clf.predict(epochs_data[test_idx])

acc = np.mean(preds == labels)
print("Classif Acc: %f " % (acc))

names = ['audio left', 'audio right', 'visual left', 'visual right']
plot_confusion_matrix(preds, labels, names)
plt.show()
# %%

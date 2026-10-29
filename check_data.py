import mne
from mne.datasets import eegbci

subject = 1
runs = [4, 8, 12]  # motor imagery: left fist vs right fist

files = eegbci.load_data(subject, runs)
raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files]
raw = mne.concatenate_raws(raws)
eegbci.standardize(raw)

print(raw.info)
print("Data shape (channels, samples):", raw.get_data().shape)
print("Sampling rate:", raw.info["sfreq"])
print("Annotations:", set(raw.annotations.description))

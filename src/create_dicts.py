import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm.contrib.itertools import product
import mne

## create the dictionary of evoked objects
eps_directory = "/Users/payamsadeghishabestari/eeg_data/teas/teas_preproc_epochs"
df_fname = Path.cwd().parent / "data" / "dataframe.csv.zip"
df = pd.read_csv(df_fname)
df = df[df["ch_name"] == "Cz"]
subjects_to_drop = [224, 305, 450, 453, 463, 488, 555, 570, 585, 946, 966, 999]
df = df.query('subject_ID not in @subjects_to_drop')
subject_ids = df["subject_ID"].unique()
stims = ['Stimulus/S 10', 'Stimulus/S 11', 'Stimulus/S 12', 'Stimulus/S 13',
        'Stimulus/S 14', 'Stimulus/S 19', 'Stimulus/S 20', 'Stimulus/S 21',
        'Stimulus/S 22', 'Stimulus/S 23', 'Stimulus/S 24', 'Stimulus/S 29']

files_list = []
for filename in sorted(os.listdir(eps_directory)): 
        f = os.path.join(eps_directory, filename)
        if (os.path.isfile(f)) and (f.endswith(".fif")) and (f[-36:-33] not in subjects_to_drop):
                files_list.append(f)

freqs = np.arange(4, 43, 3)
n_cycles = freqs / freqs[0]
evs_dict = {}
mmns_dict = {}
tfr_dict = {}
for run, cond, stim in product(["pre", "post"], ["A", "B", "C", "D"], stims):
        if run == "pre": f_run = "_1_"
        if run == "post": f_run = "_2_"
        stim_ab = f"S{stim[-2:]}"
        evs_list = []
        mmn_list = []
        tfr_list = []
        for sub_id in subject_ids:
                visit = df[(df["subject_ID"]==sub_id) & (df["run"]==run) & (df["condition"]==cond) & (df["stimulus"]==stim_ab)]["visit"].values[0]
                for fname in files_list:
                        if (f"{sub_id}" in fname) and (f_run in fname) and (f"_{visit}_" in fname):
                                ep = mne.read_epochs(fname, preload=True, verbose=False)[stim]
                                ev = ep.average()
                                evs_list.append(ev)

                                tfr = ep.compute_tfr("morlet", freqs, n_cycles=n_cycles, average=False,
                                                zero_mean=False, return_itc=False, verbose=False)
                                tfr = tfr.apply_baseline(mode="ratio", baseline=(None, 0), verbose=False).average().data[0]
                                tfr_list.append(tfr)

                                if stim in stims[1:6]:
                                        ev = mne.read_epochs(fname, preload=True, verbose=False)[stim, 'Stimulus/S 10'].average(by_event_type=True)
                                        mmn = mne.combine_evoked(ev, weights=[-1, 1])
                                if stim in stims[7:]:
                                        ev = mne.read_epochs(fname, preload=True, verbose=False)[stim, 'Stimulus/S 20'].average(by_event_type=True)
                                        mmn = mne.combine_evoked(ev, weights=[-1, 1])
                                if stim in ['Stimulus/S 10', 'Stimulus/S 20']:
                                        mmn = None
                                mmn_list.append(mmn)

                evs_dict[f"{run}_{cond}_{stim_ab}"] = evs_list
                mmns_dict[f"{run}_{cond}_{stim_ab}"] = mmn_list
                tfr_dict[f"{run}_{cond}_{stim_ab}"] = tfr_list


with open(Path.cwd().parent / "data" / "mmns.pkl", 'wb') as f:
        pickle.dump(evs_dict, f)
with open(Path.cwd().parent / "data" / "evs.pkl", 'wb') as f:
        pickle.dump(mmns_dict, f)
with open(Path.cwd().parent / "data" / "tfrs.pkl", 'wb') as f:
        pickle.dump(tfr_dict, f)
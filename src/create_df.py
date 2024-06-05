import os
from pathlib import Path
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import mne 
from scipy import stats as stats

## make dictionary of the epochs
print("making dictionary of the epochs ...")
eps_directory = "/Users/payamsadeghishabestari/eeg_data/teas/untitled folder"
conditions_file = Path.cwd().parent / "tEAS_feas_studycodes.xlsx"
df_conds = pd.read_excel(conditions_file)
verbose = False
preload = True
files_list = []
for filename in sorted(os.listdir(eps_directory)): 
        f = os.path.join(eps_directory, filename)
        if os.path.isfile(f) and f.endswith(".fif"):
                files_list.append(f)

epochs_dict = {}
for f in tqdm(files_list):
        epochs = mne.read_epochs(fname=f, preload=preload, verbose=verbose)
        epochs_dict[f"{f[-36:-33]}_{f[-24:-23]}_{f[-18:-17]}"] = epochs

## initialize the dictionary with keys (column names)
keys = ["subject_ID", "visit", "run", "condition", "stimulus", "ch_name", "baseline_variance",
        "P50_amp", "N100_amp", "P200_amp", "N200_amp", "MMN_amp_(0-450)", "MMN_amp_(100-300)", 
        "P50_lat", "N100_lat", "P200_lat", "N200_lat", "MMN_lat_(0-450)", "MMN_lat_(100-300)", 
        "P50_N100 (ptp lat)", "P50_P200 (ptp lat)", "N100_P200 (ptp lat)"]

teas_dict = {key: [] for key in keys}
times = np.linspace(epochs.tmin * 1e3, epochs.tmax * 1e3, 151)
baseline_period = [idx for idx, time in enumerate(times) if -200 <= time <= 0]
p50_idxs = [idx for idx, time in enumerate(times) if 30 <= time <= 70]
N100_idxs = [idx for idx, time in enumerate(times) if 80 <= time <= 120]
P200_idxs = [idx for idx, time in enumerate(times) if 170 <= time <= 250]
N200_idxs = [idx for idx, time in enumerate(times) if 200 <= time <= 350]
mmn_1_idxs = [idx for idx, time in enumerate(times) if 0 <= time <= 450]
mmn_2_idxs = [idx for idx, time in enumerate(times) if 100 <= time <= 300]


## filling the dictionary
print("filling out the dictionary")
for key in tqdm(list(epochs_dict.keys())):
        (subject_id, visit, run) = (key[:3], key[4:5], key[6:7]) 
        epochs = epochs_dict[key]
        for stimulus in list(epochs_dict[key].event_id.keys()):
                if "Sti" in stimulus:
                        for ch_name in epochs.info["ch_names"]:
                                data = epochs[stimulus].average().get_data(picks=ch_name)[0] * 1e6
                                
                                ## fill the dictionary 1
                                teas_dict["subject_ID"].append(subject_id)
                                teas_dict["visit"].append(visit)
                                teas_dict["run"].append(run)
                                teas_dict["stimulus"].append(stimulus)
                                teas_dict["ch_name"].append(ch_name)
                                teas_dict["baseline_variance"].append(np.var(data[baseline_period]))

                                ## fill the dictionary 2
                                teas_dict["P50_amp"].append(data[p50_idxs].max())
                                teas_dict["N100_amp"].append(data[N100_idxs].min())
                                teas_dict["P200_amp"].append(data[P200_idxs].max())
                                teas_dict["N200_amp"].append(data[N200_idxs].min())

                                ## fill the dictionary 3
                                t1 = times[p50_idxs[data[p50_idxs].argmax()]]
                                t2 = times[N100_idxs[data[N100_idxs].argmin()]]
                                t3 = times[P200_idxs[data[P200_idxs].argmax()]]
                                t4 = times[N200_idxs[data[N200_idxs].argmin()]]
                                teas_dict["P50_lat"].append(t1)
                                teas_dict["N100_lat"].append(t2)
                                teas_dict["P200_lat"].append(t3)
                                teas_dict["N200_lat"].append(t4)

                                ## fill the dictionary 4
                                if stimulus[-2:-1] == "1":
                                        mmn_data = data - epochs["Stimulus/S 10"].average().get_data(picks=ch_name)[0] * 1e6
                                if stimulus[-2:-1] == "2":
                                        mmn_data = data - epochs["Stimulus/S 20"].average().get_data(picks=ch_name)[0] * 1e6
                                
                                teas_dict["MMN_amp_(0-450)"].append(mmn_data[mmn_1_idxs].min())
                                teas_dict["MMN_amp_(100-300)"].append(mmn_data[mmn_2_idxs].min())
                                teas_dict["MMN_lat_(0-450)"].append(times[mmn_1_idxs[mmn_data[mmn_1_idxs].argmin()]])
                                teas_dict["MMN_lat_(100-300)"].append(times[mmn_2_idxs[mmn_data[mmn_2_idxs].argmin()]])

                                ## fill the dictionary 5
                                teas_dict["P50_N100 (ptp lat)"].append(t2 - t1)
                                teas_dict["P50_P200 (ptp lat)"].append(t3 - t1)
                                teas_dict["N100_P200 (ptp lat)"].append(t3 - t2)

                                ## fill the dictionary 6
                                row_idx = [i for i in range(1, len(df_conds['Unnamed: 1'])) if f"{df_conds['Unnamed: 1'][i]}"==f"{subject_id}.0"]
                                
                                match visit:
                                        case "3":
                                                condition = df_conds['Unnamed: 3'][row_idx[0]]
                                        case "4":
                                                condition = df_conds['Unnamed: 5'][row_idx[0]]
                                        case "5":
                                                condition = df_conds['Unnamed: 7'][row_idx[0]]
                                        case "6":
                                                condition = df_conds['Unnamed: 9'][row_idx[0]]

                                if len(row_idx) == 0:
                                        condition = "x"

                                teas_dict["condition"].append(condition)

df = pd.DataFrame(teas_dict) 
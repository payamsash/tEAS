import os.path as op
import copy
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import itertools
from tqdm.notebook import tqdm
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm.contrib.itertools import product
from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test


## load ##
with open(Path.cwd().parent / "data" / "mmns.pkl", 'rb') as f:
    mmns_dict = pickle.load(f)
with open(Path.cwd().parent / "data" / "evs.pkl", 'rb') as f:
    evs_dict = pickle.load(f)
with open(Path.cwd().parent / "data" / "tfrs.pkl", 'rb') as f:
    tfrs_dict = pickle.load(f)


## apply some functions ##
def fix_montage(grnd_ev):
    old_names = grnd_ev.info["ch_names"]
    new_names = []
    for ch in old_names:
        if ch == "Iz": new_names.append("FCz")
        else: new_names.append(ch)

    mapping = dict(zip(old_names, new_names))
    grnd_ev.rename_channels(mapping)
    grnd_ev.set_montage("standard_1020")

def group_elements(arr, max_diff=5):
    if len(arr) == 0:
        return []
    groups = []
    current_group = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] - arr[i-1] <= max_diff:
            current_group.append(arr[i])
        else:
            groups.append(current_group)
            current_group = [arr[i]]
    groups.append(current_group)
    return groups

for key in list(mmns_dict.keys()):
    if "S10" in key or "S20" in key:
        mmns_dict.pop(key)

## fig 1 ##
df_fname = Path.cwd().parent / "data" / "dataframe.csv.zip"
df = pd.read_csv(df_fname)
subjects_to_drop = [224, 305, 450, 453, 463, 488, 555, 570, 585, 946, 966, 999]
df = df.query('subject_ID not in @subjects_to_drop')
mapping = {"S10": "500", "S11": "500", "S12": "500", "S13": "500", "S14": "500", "S19": "500",
            "S20": "tinnitus", "S21": "tinnitus", "S22": "tinnitus", "S23": "tinnitus",
            "S24": "tinnitus", "S29": "tinnitus"}
rename_mapping = {"S20": "S10", "S21": "S11", "S22": "S12", "S23": "S13", "S24": "S14", "S29": "S19"}
df["frequency"] = df["stimulus"].map(mapping)
df['stimulus'] = df['stimulus'].replace(rename_mapping)
df = df[df["ch_name"] == "Cz"]

df1 = df.query('stimulus != "S10"')
order = ["A", "B", "C", "D"]
hue_order = ["pre", "post"]
palette_colors = ['#1f77b4', '#d62728']
g = sns.FacetGrid(data=df1, row="stimulus", col="frequency", height=11, aspect=0.6)
g.map_dataframe(sns.boxplot, x="condition", y="MMN_amp_(0-450)", hue="run", fill=False,
                hue_order=hue_order, dodge=True, linewidth=0.5, gap=0.2,
                palette=palette_colors, order=order)
g.map_dataframe(sns.stripplot, x="condition", y="MMN_amp_(0-450)", hue="run",
                hue_order=hue_order, dodge=True, size=3, palette=palette_colors)
g.set(ylim=[-6, 1])
g.savefig(Path.cwd().parent / "figures" / "1.pdf")

## fig 2 ##
fname_behave = Path.cwd().parent / "data" / "expertEAS_behav_all.xlsx"
subjects_to_drop = [224, 305, 450, 453, 463, 488, 555, 570, 585, 946, 966, 999]
my_dict = pd.read_excel(fname_behave, sheet_name=["Visit_1", "Visit_2", "Visit_3", "Visit_4"])
dfs_list = []
for key in list(my_dict.keys()): 
    df = my_dict[key]
    df.dropna(inplace=True)
    df = df.query('Subject_ID not in @subjects_to_drop')
    dfs_list.append(df)
df = pd.concat(dfs_list)

list1 = df["MML_pre"].to_list()
list2 = df["VAS_loud_pre"].to_list()
list3 = df["VAS_dist_pre"].to_list() 
list4 = [1] * len(df)

list5 = df["MML_post"].to_list()
list6 = df["VAS_loud_post"].to_list()
list7 = df["VAS_dist_post"].to_list()
list8 = [2] * len(df)
orders = [item for item in ["pre", "post"] for _ in range(len(df)*4)]
my_dict = {"subject_id": df["Subject_ID"].to_list() * 8,
            "condition": df["condition"].to_list() * 8,
            "order": orders,
            "modality": ["MMl"]*len(df) + ["VAS_L"]*len(df) + ["VAS_D"]*len(df) + ["AUC"]*len(df) + ["MMl"]*len(df) + ["VAS_L"]*len(df) + ["VAS_D"]*len(df) + ["AUC"]*len(df),
            "value": list1+list2+list3+list4+list5+list6+list7+list8}
df = pd.DataFrame(my_dict)
sub_include = [132,168,183,303,363,420,650,867,893,908,952,970,469]
df = df.query("subject_id in @sub_include")

kwargs = {"lw": 0.3, "color": "k"}
palette_colors = ['#1f77b4', '#d62728'] 
for col_name in ["MMl", "VAS_D", "VAS_L"]:
    df1 = df[df["modality"]==col_name]
    g = sns.FacetGrid(data=df1, row="condition", col="modality", height=5, aspect=0.6,
                    row_order=["A", "B", "C", "D"])
    g.map_dataframe(sns.boxplot, x="order", y="value", fill=False, gap=0.2,
                    dodge=False, linewidth=1.5, palette=palette_colors)
    g.map_dataframe(sns.stripplot, x="order", y="value", size=4,
                    palette=palette_colors)
    
# figure 2 and 3 (ev + mmn + 2 factors)
pick = "Cz"
conds = ["A", "B", "C", "D"]
stims = ["S11", "S12", "S13", "S14", "S19", "S21", "S22", "S23", "S24", "S29"]
for cond, stim in product(conds, stims):
    p_thr = 0.05
    factor_levels = [2, 1]
    my_dict = copy.deepcopy(mmns_dict) # or mmns_dict

    for key in list(my_dict.keys()):
        for item in my_dict[key]:
            item.filter(0.1, 30, picks=pick, verbose=False)
            item.crop(tmin=-0.05, tmax=0.450)    

    data_pre = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"pre_{cond}_{stim}"]])
    data_post = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"post_{cond}_{stim}"]])
    data = np.swapaxes([data_pre, data_post], axis1=0, axis2=1)
    f_vals, p_vals = mne.stats.f_mway_rm(data=data, factor_levels=factor_levels, effects="all", correction=False)
    fvals = f_vals[0]
    pvals = p_vals[0]
    times = np.linspace(-50, 450, len(pvals))
    grnd_ev_pre = mne.grand_average(my_dict[f"pre_{cond}_{stim}"])
    grnd_ev_post = mne.grand_average(my_dict[f"post_{cond}_{stim}"])
    grnd_ev_st = mne.grand_average(evs_dict[f"post_{cond}_S10"]).crop(tmin=-0.05, tmax=0.450) 
    grnd_ev_pre_data = grnd_ev_pre.get_data(picks=pick)[0] * 1e6 
    grnd_ev_post_data = grnd_ev_post.get_data(picks=pick)[0] * 1e6 
    sem_pre = sem(data_pre, axis=0) * 1e6
    sem_post = sem(data_post, axis=0) * 1e6
    sub_times = times[np.where(pvals < p_thr)[0]]
    # sub_fvals = fvals[np.where(pvals < p_thr)[0]]
    mask = pvals < p_thr
    convolved = np.convolve(mask, np.ones(3, dtype=int), 'same') >= 2
    result_indices = np.where(mask & convolved)[0]
    sub_fvals = fvals[result_indices]

    # plotting
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7, as_cmap=True)
    cl_dict_ev = {"pre": tuple(pal.colors[0]), "post": tuple(pal.colors[170]), "standard": "k"}
    styles={"pre": {"linewidth": 2.5}, "post": {"linewidth": 2.5}, "standard": {"linewidth": 1.5, "linestyle": "--"}}

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    mne.viz.plot_compare_evokeds({"pre": grnd_ev_pre, "post": grnd_ev_post, "standard": grnd_ev_st},
                                picks=pick, legend=True,
                                truncate_xaxis=False, truncate_yaxis=False, time_unit='ms', axes=axs[0],
                                colors=cl_dict_ev, show_sensors=False, styles=styles,
                                title=f" Condition: {cond}   Stimulus: {stim}")
    for ev_data, sem_d, color_id in zip([grnd_ev_pre_data, grnd_ev_post_data], [sem_pre, sem_post], [0, 170]):
        axs[0].fill_between(times, ev_data - sem_d,
                            ev_data + sem_d, alpha=0.2,
                            color=pal.colors[color_id])

    # axs[1].plot(times, fvals, color="#1f77b4", lw=2)
    axs[1].plot(times, fvals, color="#1f77b4", lw=2)
    for sub_group in group_elements(result_indices, 3):
        if len(sub_group) > 4:
            axs[1].plot(times[sub_group], fvals[sub_group], color='red', label='x > threshold for at least 3 samples')
    axs[1].set_xlim([-50, 450])
    axs[1].spines[["top", "right"]].set_visible(False)

    # for idx, i in enumerate(np.diff(sub_times)):
    #     if i < 5:
    #         axs[1].plot(sub_times[idx:idx+2], sub_fvals[idx:idx+2], color="#d62728", lw=2)
            
    [axs[i].axvspan(xmin=-200, xmax=0, color="grey", alpha=0.1) for i in range(2)]
    axs[1].vlines(x=0, ymin=0, ymax=max(fvals), colors="k", linestyles="dashed")
    axs[1].set_ylabel("F vlaues")
    axs[0].set_xlabel("")
    axs[1].set_xlabel("Time (ms)")
    axs[0].legend(loc="upper right", frameon=False)

    fig.savefig(Path.cwd().parent / "figures_30" / "mmns" /f"f_mmn_{cond}_{stim}.pdf")

### topo plots
conds = ["A", "B", "C", "D"]
stims = ["S11", "S12", "S13", "S14", "S19", "S21", "S22", "S23", "S24", "S29"]
for cond, stim in itertools.product(conds, stims):
    grnd_ev_pre = mne.grand_average(evs_dict[f"pre_{cond}_{stim}"])
    grnd_ev_post = mne.grand_average(evs_dict[f"post_{cond}_{stim}"])
    grnd_mmn_pre = mne.grand_average(mmns_dict[f"pre_{cond}_{stim}"])
    grnd_mmn_post = mne.grand_average(mmns_dict[f"post_{cond}_{stim}"])
    [fix_montage(grnd_ev) for grnd_ev in [grnd_ev_pre, grnd_ev_post, grnd_mmn_pre, grnd_mmn_post]]

    fig, axs = plt.subplots(2, 4, figsize=(7, 4), layout="constrained")
    times = [0.05, 0.11, 0.23, 0.275]
    avgs = [0.03, 0.06, 0.06, 0.35]
    vlims = [(-1.1, 1.1), (-3.3, 3.3), (-1.8, 1.8), (-1.1, 1.1)]
    for idx, time, avg, vlim, ax in zip(range(4), times, avgs, vlims, axs[0]):
        grnd_ev_pre.plot_topomap(times=time, average=avg, vlim=vlim,
                                    size=5, colorbar=False, time_unit="ms",
                                    axes=ax)
        if idx == 3:
            grnd_mmn_pre.plot_topomap(times=time, average=avg, vlim=vlim,
                                    size=5, colorbar=False, time_unit="ms",
                                    axes=ax)

    for idx, time, avg, vlim, ax in zip(range(4),times, avgs, vlims, axs[1]):
        grnd_ev_post.plot_topomap(times=time, average=avg, vlim=vlim,
                                    size=5, colorbar=False, time_unit="ms",
                                    axes=ax)
        if idx == 3:
            grnd_mmn_post.plot_topomap(times=time, average=avg, vlim=vlim,
                                    size=5, colorbar=False, time_unit="ms",
                                    axes=ax)
    fig.savefig(Path.cwd().parent / "figures" / f"topo_{cond}_{stim}.pdf")

## mmn + 4 factors
pick = "Cz"
conds = ["A", "B", "D"]
stims = ["S11", "S12", "S13", "S14", "S19", "S21", "S22", "S23", "S24", "S29"]
for cond1, cond2 in product(conds, conds):
    if not cond1 == cond2:

        for stim in tqdm(stims[:2]):
            p_thr = 0.05
            factor_levels = [2, 2]
            my_dict = copy.deepcopy(mmns_dict)
            for key in list(my_dict.keys()):
                for item in my_dict[key]:
                    item.filter(0.1, 30, picks=pick, verbose=False)
                    item.crop(tmin=-0.05, tmax=0.450)    

            data_pre_1 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"pre_{cond1}_{stim}"]])
            data_pre_2 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"pre_{cond2}_{stim}"]])
            data_post_1 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"post_{cond1}_{stim}"]])
            data_post_2 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"post_{cond2}_{stim}"]])

            data = np.swapaxes([data_pre_1, data_pre_2, data_post_1, data_post_2], axis1=0, axis2=1)
            f_vals, p_vals = mne.stats.f_mway_rm(data=data, factor_levels=factor_levels, effects="all", correction=True)
            fvals = f_vals[-1]
            pvals = p_vals[-1]
            times = np.linspace(-50, 450, len(pvals))

            grnd_ev_pre_1 = mne.grand_average(my_dict[f"pre_{cond1}_{stim}"])
            grnd_ev_post_1 = mne.grand_average(my_dict[f"post_{cond1}_{stim}"])
            grnd_ev_pre_2 = mne.grand_average(my_dict[f"pre_{cond2}_{stim}"])
            grnd_ev_post_2 = mne.grand_average(my_dict[f"post_{cond2}_{stim}"])

            grnd_ev_pre_1_data = grnd_ev_pre_1.get_data(picks=pick)[0] * 1e6 
            grnd_ev_post_1_data = grnd_ev_post_1.get_data(picks=pick)[0] * 1e6 
            grnd_ev_pre_2_data = grnd_ev_pre_2.get_data(picks=pick)[0] * 1e6 
            grnd_ev_post_2_data = grnd_ev_post_2.get_data(picks=pick)[0] * 1e6 

            sem_pre_1 = sem(data_pre_1, axis=0) * 1e6
            sem_post_1 = sem(data_post_1, axis=0) * 1e6
            sem_pre_2 = sem(data_pre_2, axis=0) * 1e6
            sem_post_2 = sem(data_post_2, axis=0) * 1e6

            sub_times = times[np.where(pvals < p_thr)[0]]
            # sub_fvals = fvals[np.where(pvals < p_thr)[0]]

            mask = pvals < p_thr
            convolved = np.convolve(mask, np.ones(3, dtype=int), 'same') >= 2
            result_indices = np.where(mask & convolved)[0]
            sub_fvals = fvals[result_indices]

            # plotting
            pal_1 = sns.cubehelix_palette(10, rot=-.25, light=.7, as_cmap=True)
            pal_2 = sns.cubehelix_palette(10, rot=1, light=.3, as_cmap=True)

            cl_dict_ev = {"pre_1": tuple(pal_1.colors[0]), "post_1": tuple(pal_1.colors[128]), "pre_2": tuple(pal_2.colors[0]), "post_2": tuple(pal_2.colors[250])}
            styles={"pre_1": {"linewidth": 2.5}, "post_1": {"linewidth": 2.5}, "pre_2": {"linewidth": 2.5}, "post_2": {"linewidth": 2.5}}

            fig, axs = plt.subplots(2, 1, figsize=(8, 6))
            mne.viz.plot_compare_evokeds({"pre_1": grnd_ev_pre_1, "post_1": grnd_ev_post_1, "pre_2": grnd_ev_pre_2, "post_2": grnd_ev_post_2},
                                        picks=pick, legend=True,
                                        truncate_xaxis=False, truncate_yaxis=False, time_unit='ms', axes=axs[0],
                                        colors=cl_dict_ev, show_sensors=False, styles=styles,
                                        title=f" Condition: {cond1} vs {cond2} &  Stimulus: {stim}")
            # for ev_data, sem_d, color_id in zip([grnd_ev_pre_1_data, grnd_ev_post_1_data, grnd_ev_pre_2_data, grnd_ev_post_2_data], [sem_pre_1, sem_post_1, sem_pre_2, sem_post_2], [0, 60, 170, 240]):
            #     axs[0].fill_between(times, ev_data - sem_d,
            #                         ev_data + sem_d, alpha=0.2,
            #                         color=pal.colors[color_id])

            axs[1].plot(times, fvals, color="#1f77b4", lw=2)
            axs[1].set_xlim([-50, 450])
            axs[1].spines[["top", "right"]].set_visible(False)
            for sub_group in group_elements(result_indices, 3):
                if len(sub_group) > 4:
                    axs[1].plot(times[sub_group], fvals[sub_group], color='red', label='x > threshold for at least 3 samples')

            # for idx, i in enumerate(np.diff(sub_times)):
            #     if i < 5:
            #         axs[1].plot(sub_times[idx:idx+2], sub_fvals[idx:idx+2], color="#d62728", lw=2)
            
            [axs[i].axvspan(xmin=-200, xmax=0, color="grey", alpha=0.1) for i in range(2)]
            axs[1].vlines(x=0, ymin=0, ymax=max(fvals), colors="k", linestyles="dashed")
            axs[1].set_ylabel("F vlaues")
            axs[0].set_xlabel("")
            axs[1].set_xlabel("Time (ms)")
            axs[0].legend(loc="upper right", frameon=False)

            fig.savefig(Path.cwd().parent / "figures_30" / "mmn_4_factors" /f"f_mmn_{cond1}_{cond2}_{stim}.pdf")

## source computation ##
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
grnd_ev = mne.grand_average(evs_dict["pre_A_S10"])
fix_montage(grnd_ev)
info = grnd_ev.info
fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None, verbose=False)
noise_cov = mne.make_ad_hoc_cov(info, std=None, verbose=False)
inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2


conds = ["A", "B", "C", "D"]
stims = ["S11", "S12", "S13", "S14"]

stc_evs_dict = {}
stc_mmns_dict = {}

for cond, stim in itertools.product(conds, stims):
    print(stim)
    grnd_ev_pre = mne.grand_average(evs_dict[f"pre_{cond}_{stim}"])
    grnd_ev_post = mne.grand_average(evs_dict[f"post_{cond}_{stim}"])
    grnd_mmn_pre = mne.grand_average(mmns_dict[f"pre_{cond}_{stim}"])
    grnd_mmn_post = mne.grand_average(mmns_dict[f"post_{cond}_{stim}"])
    [fix_montage(grnd_ev) for grnd_ev in [grnd_ev_pre, grnd_ev_post, grnd_mmn_pre, grnd_mmn_post]]
    [grnd_ev.set_eeg_reference("average", projection=True) for grnd_ev in [grnd_ev_pre, grnd_ev_post, grnd_mmn_pre, grnd_mmn_post]]
    
    stc_evs_dict[f"pre_{cond}_{stim}"] = apply_inverse(grnd_ev_pre, inverse_operator, lambda2, method=method,
                        pick_ori=None, return_residual=False, verbose=False).crop(tmin=0.084, tmax=0.140).mean()
    stc_evs_dict[f"post_{cond}_{stim}"] = apply_inverse(grnd_ev_post, inverse_operator, lambda2, method=method,
                        pick_ori=None, return_residual=False, verbose=False).crop(tmin=0.084, tmax=0.140).mean()
    stc_mmns_dict[f"pre_{cond}_{stim}"] = apply_inverse(grnd_mmn_pre, inverse_operator, lambda2, method=method,
                        pick_ori=None, return_residual=False, verbose=False).crop(tmin=0.104, tmax=0.448).mean()
    stc_mmns_dict[f"post_{cond}_{stim}"] = apply_inverse(grnd_mmn_post, inverse_operator, lambda2, method=method,
                        pick_ori=None, return_residual=False, verbose=False).crop(tmin=0.104, tmax=0.448).mean()
    
## source plots ## 
colormap = "RdBu" # coolwarm for ev
# clim = dict(kind="value", lims=[35, 40, 45]) # for ev
clim = dict(kind="value", lims=[10, 15, 20]) # for mmn

for cond, stim in itertools.product(conds, stims):
    brain_pre = stc_mmns_dict[f"pre_{cond}_{stim}"].plot(views="lat", surface='inflated', hemi="split", size=(800, 400),
                                    subject="fsaverage", subjects_dir=subjects_dir,
                                    background="w", colorbar=False, clim=clim,  colormap=colormap,
                                    time_viewer=False, show_traces=False).screenshot()
    brain_post = stc_mmns_dict[f"post_{cond}_{stim}"].plot(views="lat", surface='inflated', hemi="split", size=(800, 400),
                                    subject="fsaverage", subjects_dir=subjects_dir,
                                    background="w", colorbar=False, clim=clim,  colormap=colormap,
                                    time_viewer=False, show_traces=False).screenshot()

    fig = plt.figure(figsize=(9, 7))
    axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)
    for brain, ax, title in zip([brain_pre, brain_post], axes, ["pre", "post"]):
        nonwhite_pix = (brain != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = brain[nonwhite_row][:, nonwhite_col]
        ax.imshow(cropped_screenshot)
        ax.axis("off")
        ax.set_title(f"{title}")
    fig.savefig(Path.cwd().parent / "figures" / "source" /f"mmn_{cond}_{stim}.pdf")

## latencies ##
fname = Path.cwd().parent / "data" / "dataframe.csv.zip"
df = pd.read_csv(fname)

palette_colors = ['#1f77b4', '#d62728'] 
df1 = df.query('ch_name == "Cz" & stimulus == "S12"')
fig, axs = plt.subplots(3, 1, figsize=(13, 10))
sns.boxplot(data=df1, x="condition", y="P50_lat", hue="run", fill=True, gap=0.2, order=["A", "B", "C", "D"],
                    dodge=True, linewidth=1.5, palette=palette_colors, ax=axs[0])
sns.boxplot(data=df1, x="condition", y="N100_lat", hue="run", fill=True, gap=0.2, order=["A", "B", "C", "D"],
                    dodge=True, linewidth=1.5, palette=palette_colors, ax=axs[1])
sns.boxplot(data=df1, x="condition", y="P200_lat", hue="run", fill=True, gap=0.2, order=["A", "B", "C", "D"],
                    dodge=True, linewidth=1.5, palette=palette_colors, ax=axs[2])



##### Source Plots #####
condition = "B"
stim = "S11"
gev_1 = mne.grand_average(evs_dict[f"pre_{condition}_{stim}"])
gev_2 = mne.grand_average(evs_dict[f"post_{condition}_{stim}"])
gev_diff = mne.combine_evoked(all_evoked=[gev_1, gev_2], weights=[-1, 1])

## source computation ##
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src_fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(src_fname)
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
fix_montage(gev_diff)
info = gev_diff.info
fwd = mne.make_forward_solution(info, trans=trans, src=src_fname, bem=bem, eeg=True, mindist=5.0, n_jobs=None, verbose=False)
noise_cov = mne.make_ad_hoc_cov(info, std=None, verbose=False)
inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2
gev_diff.set_eeg_reference("average", projection=True)
stc = apply_inverse(gev_diff, inverse_operator, lambda2, method=method,
                        pick_ori=None, return_residual=False, verbose=False)
stc_sub = stc.copy().crop(0.090, 0.110)
stc_mean = stc_sub.mean()
# stc_sub = stc.copy().crop(0.195, 0.205)
# stc_mean = stc_sub.mean()

brain_kwargs = dict(background="white", surf="pial_semi_inflated", cortex=["#b8b4ac", "#b8b4ac"])
clim_dict = {"kind": "value", "lims": [7, 11.5, 16]}
kwargs = {"surface" : "pial_semi_inflated",
    "size": (800, 400),
    "subject": "fsaverage",
    "subjects_dir": None,
    "initial_time": 0.1,
    "background": "w",
    "colorbar": False,
    "clim": clim_dict,
    "colormap": "hot",
    "time_viewer": False,
    "show_traces": False,
    "alpha": 0.7,
    "brain_kwargs":brain_kwargs}

brain_lh_lat = stc_mean.plot(views="lat", hemi="lh", **kwargs).screenshot()
brain_rh_lat = stc_mean.plot(views="lat", hemi="rh", **kwargs).screenshot()
brain_lh_med = stc_mean.plot(views="medial", hemi="lh", **kwargs).screenshot()
brain_rh_med = stc_mean.plot(views="medial", hemi="rh", **kwargs).screenshot()
brain_rostral = stc_mean.plot(views="rostral", hemi="both", **kwargs).screenshot()
brain_ventral = stc_mean.plot(views="ventral", hemi="both", **kwargs).screenshot()

fig, axes = plt.subplots(2, 2, figsize=(9, 6))
fig.subplots_adjust(hspace=0.1)
for ax, brain in zip([axes[0][0], axes[0][1], axes[1][0], axes[1][1]],
                    [brain_lh_med, brain_rh_med, brain_lh_lat, brain_rh_lat]):
    nonwhite_pix = (brain != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = brain[nonwhite_row][:, nonwhite_col]
    ax.imshow(cropped_screenshot)
    ax.axis("off")
fig.tight_layout()
fig.savefig(Path.cwd().parent / "figures" / "brain_200.pdf")
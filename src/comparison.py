from pathlib import Path
import pickle
import numpy as np
import scipy
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable


def compare_pre_post_one_factor(data, type, cond, stim, pick):
    """

    Parameters
    ----------
    data : dict
        Either dictionary of evoked responses or MMN. 
    type : str
        If MMN, change type to "mmn" else set it to "ev".
    cond : str
        Condition name.
    stim : str
        Stimulus name like "S11".
    pick : str
        Channel name like "Cz".

    """

    p_thr = 0.05
    factor_levels = [2, 1]
    my_dict = data
    data_pre = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"pre_{cond}_{stim}"]])
    data_post = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"post_{cond}_{stim}"]])
    data = np.swapaxes([data_pre, data_post], axis1=0, axis2=1)
    f_vals, p_vals = mne.stats.f_mway_rm(data=data, factor_levels=factor_levels, effects="all", correction=True)
    fvals = f_vals[0]
    pvals = p_vals[0]
    times = np.linspace(-200, 500, len(pvals))
    grnd_ev_pre = mne.grand_average(my_dict[f"pre_{cond}_{stim}"])
    grnd_ev_post = mne.grand_average(my_dict[f"post_{cond}_{stim}"])
    grnd_ev_pre_data = grnd_ev_pre.get_data(picks=pick)[0] * 1e6 
    grnd_ev_post_data = grnd_ev_post.get_data(picks=pick)[0] * 1e6 
    sem_pre = sem(data_pre, axis=0) * 1e6
    sem_post = sem(data_post, axis=0) * 1e6
    sub_times = times[np.where(pvals < p_thr)[0]]
    sub_fvals = fvals[np.where(pvals < p_thr)[0]]

    # plotting
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7, as_cmap=True)
    cl_dict_ev = {"pre": tuple(pal.colors[0]), "post": tuple(pal.colors[170])}
    styles={"pre": {"linewidth": 2.5}, "post": {"linewidth": 2.5}}

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    mne.viz.plot_compare_evokeds({"pre": grnd_ev_pre, "post": grnd_ev_post}, picks=pick, legend=True,
                                time_unit='ms', axes=axs[0], colors=cl_dict_ev, show_sensors=False,
                                styles=styles, title=f"type: {type}   Condition: {cond}   Stimulus: {stim}")
    for ev_data, sem_d, color_id in zip([grnd_ev_pre_data, grnd_ev_post_data], [sem_pre, sem_post], [0, 170]):
        axs[0].fill_between(times, ev_data - sem_d,
                            ev_data + sem_d, alpha=0.2,
                            color=pal.colors[color_id])

    axs[1].plot(times, fvals, color="#1f77b4", lw=2)
    axs[1].set_xlim([-200, 500])
    axs[1].spines[["top", "right"]].set_visible(False)

    for idx, i in enumerate(np.diff(sub_times)):
        if i < 5:
            axs[1].plot(sub_times[idx:idx+2], sub_fvals[idx:idx+2], color="#d62728", lw=2)

    [axs[i].axvspan(xmin=-200, xmax=0, color="grey", alpha=0.1) for i in range(2)]
    axs[1].vlines(x=0, ymin=0, ymax=max(fvals), colors="k", linestyles="dashed")
    axs[1].set_ylabel("F vlaues")
    axs[0].set_xlabel("")
    axs[1].set_xlabel("Time (ms)")


def compare_pre_post_two_factors(data, type, conds, stims, pick):
    """

    Parameters
    ----------
    data : dict
        Either dictionary of evoked responses or MMN. 
    type : str
        If MMN, change type to "mmn" else set it to "ev".
    conds : list
        Condition names e.g. ["B", "D"].
    stims : list
        Stimulus name like ["S11", "S12"].
    pick : str
        Channel name like "Cz".

    Note: Only one of stims or conds parameter should have len 2.
    """

    my_dict = data
    p_thr = 0.05
    factor_levels = [2, 2]

    if len(conds) == 2:
        data_1 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"pre_{conds[0]}_{stims[0]}"]])
        data_2 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"pre_{conds[1]}_{stims[0]}"]])
        data_3 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"post_{conds[0]}_{stims[0]}"]])
        data_4 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"post_{conds[1]}_{stims[0]}"]])

        grnd_ev_1 = mne.grand_average(my_dict[f"pre_{conds[0]}_{stims[0]}"])
        grnd_ev_2 = mne.grand_average(my_dict[f"pre_{conds[1]}_{stims[0]}"])
        grnd_ev_3 = mne.grand_average(my_dict[f"post_{conds[0]}_{stims[0]}"])
        grnd_ev_4 = mne.grand_average(my_dict[f"post_{conds[1]}_{stims[0]}"])
        title = f"type: {type}   pre/post   {conds[0]}/{conds[1]}"

    if len(stims) == 2:
        data_1 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"pre_{conds[0]}_{stims[0]}"]])
        data_2 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"pre_{conds[0]}_{stims[1]}"]])
        data_3 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"post_{conds[0]}_{stims[0]}"]])
        data_4 = np.squeeze([ev.get_data(picks=pick) for ev in my_dict[f"post_{conds[0]}_{stims[1]}"]])

        grnd_ev_1 = mne.grand_average(my_dict[f"pre_{conds[0]}_{stims[0]}"])
        grnd_ev_2 = mne.grand_average(my_dict[f"pre_{conds[0]}_{stims[1]}"])
        grnd_ev_3 = mne.grand_average(my_dict[f"post_{conds[0]}_{stims[0]}"])
        grnd_ev_4 = mne.grand_average(my_dict[f"post_{conds[0]}_{stims[1]}"])
        title = f"type: {type}   pre/post   {stims[0]}/{stims[1]}"

    data = np.swapaxes([data_1, data_2, data_3, data_4], axis1=0, axis2=1)
    f_vals, p_vals = mne.stats.f_mway_rm(data=data, factor_levels=factor_levels, effects="all", correction=True)
    fvals = f_vals[-1]
    pvals = p_vals[-1]
    times = np.linspace(-200, 500, len(fvals))
    sub_times = times[np.where(pvals < p_thr)[0]]
    sub_fvals = fvals[np.where(pvals < p_thr)[0]]

    # plotting
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7, as_cmap=True)
    cl_dict_ev = {"pre_1": tuple(pal.colors[0]), "pre_2": tuple(pal.colors[64]),
                "post_1": tuple(pal.colors[128]), "post_2": tuple(pal.colors[255])}
    styles={"pre_1": {"linewidth": 2.5}, "pre_2": {"linewidth": 2.5},
            "post_1": {"linewidth": 2.5}, "post_2": {"linewidth": 2.5}}

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    mne.viz.plot_compare_evokeds({"pre_1": grnd_ev_1, "pre_2": grnd_ev_2, "post_1": grnd_ev_3, "post_2": grnd_ev_4},
                                picks=pick, legend=True, time_unit='ms', axes=axs[0], colors=cl_dict_ev, show_sensors=True,
                                styles=styles, title=title)
    axs[1].plot(times, fvals, color="#1f77b4", lw=2)
    axs[1].set_xlim([-200, 500])
    axs[1].spines[["top", "right"]].set_visible(False)

    for idx, i in enumerate(np.diff(sub_times)):
        if i < 5:
            axs[1].plot(sub_times[idx:idx+2], sub_fvals[idx:idx+2], color="#d62728", lw=2)

    [axs[i].axvspan(xmin=-200, xmax=0, color="grey", alpha=0.1) for i in range(2)]
    axs[1].vlines(x=0, ymin=0, ymax=max(fvals), colors="k", linestyles="dashed")
    axs[1].set_ylabel("F vlaues")
    axs[0].set_xlabel("")
    axs[1].set_xlabel("Time (ms)")


def compare_pre_post_one_factor_tfr(data, cond, stim):
    """

    Parameters
    ----------
    data : dict
        Either dictionary of evoked responses or MMN. 
    type : str
        If MMN, change type to "mmn" else set it to "ev".
    cond : str
        Condition name.
    stim : str
        Stimulus name like "S11".
    pick : str
        Channel name like "Cz".

    """
    my_dict = data
    p_thr = 0.05
    extent = [-200, 500, 4, 43]
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7, as_cmap=True)

    factor_levels = [2, 1]
    data_pre = np.array(my_dict[f"pre_{cond}_{stim}"])
    data_post = np.array(my_dict[f"post_{cond}_{stim}"])
    data = np.swapaxes([data_pre, data_post], axis1=0, axis2=1)
    f_vals, p_vals = mne.stats.f_mway_rm(data=data, factor_levels=factor_levels, effects="all", correction=False)
    fvals = f_vals[0]
    pvals = p_vals[0]
    fig, (ax, ax1) = plt.subplots(2, 1, layout="constrained")
    c= ax.imshow(fvals, cmap="gray", aspect="auto", origin="lower", extent=extent)

    fvals[pvals >= p_thr] = np.nan
    c = ax.imshow(fvals, cmap=pal, aspect="auto", origin="lower", extent=extent)
    fig.colorbar(c, ax=ax)
    # ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Condition: {cond}   Stim: {stim}   not corrected")

    ## multiple comparison with FDR
    def stat_fun(*args):
        return mne.stats.f_mway_rm(
            np.swapaxes(args, 1, 0),
            factor_levels=factor_levels,
            effects="A",
            return_pvals=False,
        )[0]
    pthresh = 0.001  # set threshold rather high to save some time
    f_thresh = mne.stats.f_threshold_mway_rm(13, factor_levels, "A", pthresh)
    tail = 1 
    n_permutations = 1000
    F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
        list(np.swapaxes(data, axis1=0, axis2=1)),
        stat_fun=stat_fun,
        threshold=f_thresh, # lower thr make more clusters
        tail=tail,
        n_jobs=None,
        n_permutations=n_permutations,
        buffer_size=None,
        out_type="mask",
    )
    good_clusters = np.where(cluster_p_values < p_thr)[0]
    if len(good_clusters) > 0:
        F_obs_plot = F_obs.copy()
        F_obs_plot[~clusters[np.squeeze(good_clusters)]] = np.nan

        for f_image, cmap in zip([F_obs, F_obs_plot], ["gray", pal]):
            c = ax1.imshow(
                f_image,
                cmap=cmap,
                aspect="auto",
                origin="lower",
                extent=extent,
            )
        fig.colorbar(c, ax=ax1)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Frequency (Hz)")


def compare_pre_post_two_factors_tfr(data, conds, stims):
    """

    Parameters
    ----------
    data : dict
        Either dictionary of evoked responses or MMN. 
    type : str
        If MMN, change type to "mmn" else set it to "ev".
    conds : list
        Condition names e.g. ["B", "D"].
    stims : list
        Stimulus name like ["S11", "S12"].
    pick : str
        Channel name like "Cz".

    Note: Only one of stims or conds parameter should have len 2.
    """
    my_dict = data
    extent = [-200, 500, 4, 43]
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7, as_cmap=True)
    p_thr = 0.05
    factor_levels = [2, 2]
    if len(conds) == 2:
        
        data_1 = np.array(my_dict[f"pre_{conds[0]}_{stims[0]}"])
        data_2 = np.array(my_dict[f"pre_{conds[1]}_{stims[0]}"])
        data_3 = np.array(my_dict[f"post_{conds[0]}_{stims[0]}"])
        data_4 = np.array(my_dict[f"post_{conds[1]}_{stims[0]}"])
        title = f"pre/post & {conds[0]}/{conds[1]} & {stims[0]}"

    if len(stims) == 2:
        data_1 = np.array(my_dict[f"pre_{conds[0]}_{stims[0]}"])
        data_2 = np.array(my_dict[f"pre_{conds[0]}_{stims[1]}"])
        data_3 = np.array(my_dict[f"post_{conds[0]}_{stims[0]}"])
        data_4 = np.array(my_dict[f"post_{conds[0]}_{stims[1]}"])
        title = f"pre/post & {stims[0]}/{stims[1]} & {conds[0]}"

    data = np.swapaxes([data_1, data_2, data_3, data_4], axis1=0, axis2=1)
    f_vals, p_vals = mne.stats.f_mway_rm(data=data, factor_levels=factor_levels, effects="all", correction=False)
    fvals = f_vals[-1]
    pvals = p_vals[-1]
    fig, (ax, ax1) = plt.subplots(2, 1, layout="constrained")
    c= ax.imshow(fvals, cmap="gray", aspect="auto", origin="lower", extent=extent)

    fvals[pvals >= p_thr] = np.nan
    c = ax.imshow(fvals, cmap=pal, aspect="auto", origin="lower", extent=extent)
    fig.colorbar(c, ax=ax)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    ## multiple comparison with FDR
    def stat_fun(*args):
        return mne.stats.f_mway_rm(
            np.swapaxes(args, 1, 0),
            factor_levels=factor_levels,
            effects="A:B",
            return_pvals=False,
        )[0]
    pthresh = 0.001  # set threshold rather high to save some time
    f_thresh = mne.stats.f_threshold_mway_rm(13, factor_levels, "A:B", pthresh)
    tail = 1 
    n_permutations = 1000
    F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
        list(np.swapaxes(data, axis1=0, axis2=1)),
        stat_fun=stat_fun,
        threshold=None, # lower thr make more clusters
        tail=tail,
        n_jobs=None,
        n_permutations=n_permutations,
        buffer_size=None,
        out_type="mask",
    )
    good_clusters = np.where(cluster_p_values < p_thr)[0]
    if len(good_clusters) > 0:
        F_obs_plot = F_obs.copy()
        F_obs_plot[~clusters[np.squeeze(good_clusters)]] = np.nan

        for f_image, cmap in zip([F_obs, F_obs_plot], ["gray", pal]):
            c = ax1.imshow(
                f_image,
                cmap=cmap,
                aspect="auto",
                origin="lower",
                extent=extent,
            )
        fig.colorbar(c, ax=ax1)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Frequency (Hz)")



def spatiotemporal_perm_f_test(data, type, conds, stim, p_accept):
    """

    Parameters
    ----------
    data : dict
        Dictionary of evoked responses. 
    type : str
        Either "2_factor" or "4_factor".
    conds : list | str
        Condition names e.g. ["B", "D"].
    stim : str
        Stimulus name like "S11".

    Note: Only one of stims or conds parameter should have len 2.
    """
    my_dict = data
    info = my_dict[f"pre_A_S11"][0].info
    
    if type == "2_factor":
        data_pre = np.array([ev.get_data() for ev in my_dict[f"pre_{conds}_{stim}"]])
        data_post = np.array([ev.get_data() for ev in my_dict[f"post_{conds}_{stim}"]])
        X = np.array([data_pre, data_post])

    if type == "4_factor":
        data_1 = np.array([ev.get_data() for ev in my_dict[f"pre_{conds[0]}_{stim}"]])
        data_2 = np.array([ev.get_data() for ev in my_dict[f"pre_{conds[1]}_{stim}"]])
        data_3 = np.array([ev.get_data() for ev in my_dict[f"post_{conds[0]}_{stim}"]])
        data_4 = np.array([ev.get_data() for ev in my_dict[f"post_{conds[1]}_{stim}"]])
        X = np.array([data_1, data_2, data_3, data_4])

    X = list(np.swapaxes(X, axis1=2, axis2=3))

    adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    tail = 1
    alpha_cluster_forming = 0.001
    n_conditions = len(X)
    n_observations = len(X[0])
    dfn = n_conditions - 1
    dfd = n_observations - n_conditions
    f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)
    cluster_stats = mne.stats.spatio_temporal_cluster_test(X, n_permutations=1000,
                                                threshold=None, # smaller thr not the f_thresh
                                                tail=tail, n_jobs=None,
                                                buffer_size=None,
                                                adjacency=adjacency)
    F_obs, clusters, p_values, _ = cluster_stats
    good_cluster_inds = np.where(p_values < p_accept)[0]
    print(p_values)

    if len(good_cluster_inds) > 0:
        fig, ax_topos = plt.subplots(1, len(good_cluster_inds), figsize=(10, 3))
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            time_inds, space_inds = np.squeeze(clusters[clu_idx])
            ch_inds = np.unique(space_inds)
            time_inds = np.unique(time_inds)
            f_map = F_obs[time_inds, ...].mean(axis=0)
            sig_times = my_dict[f"pre_A_S11"][0].times[time_inds]
            mask = np.zeros((f_map.shape[0], 1), dtype=bool)
            mask[ch_inds, :] = True

            f_evoked = mne.EvokedArray(f_map[:, np.newaxis], info, tmin=0)
            f_evoked.plot_topomap(
                times=0,
                mask=mask,
                axes=ax_topos[i_clu],
                cmap="Reds",
                vlim=(np.min, np.max),
                show=False,
                colorbar=False,
                mask_params=dict(markersize=10),
            )
            image = ax_topos[i_clu].images[0]
            ax_topos[i_clu].set_title("")
            divider = make_axes_locatable(ax_topos[i_clu])
            ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(image, cax=ax_colorbar)
            ax_topos[i_clu].set_xlabel(
                "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
            )

        plt.show()
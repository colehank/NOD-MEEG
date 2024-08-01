#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne,os
from scipy.stats import zscore
import json
import sys
import os.path as op
sys.path.append(op.abspath('..'))
# import utils.MEG as nod
#%% nc for ImageNet(by class)
names                       = ['O','T','P','F','C']
labs                        = ['Occipital','Temporal','Parietal','Frontal','Central']
colors                      = plt.get_cmap('tab10')
epo_root                    = '/nfs/z1/userhome/zzl-zhangguohao/BrainImageNet/NaturalObject/MEG/NOD/derivatives/epochs'
fig_dir                     = '../results'
fontsize_other              = 12
fontsize_other              = 10
subs                        = [f'{s:02d}' for s in range(1,31)]
n_rep                       = 4
n_stim                      = 1000
colors                      = plt.get_cmap('tab10')



#%%
def all_epos(epo_root,subs):
    all_epochs = {}
    for sub in subs:
        print(f'\rloading sub-{sub}',end = '',flush=True)
        
        epochs = mne.read_epochs(f'{epo_root}/preprocessed_sub-{sub}_task-ImageNet_epo.fif')
        all_epochs[sub] = epochs
    return all_epochs

def unique_stim_epochs(epochs, n=1):
    """
    返回第n个的stimulus的epoch
    """
    metadata = epochs.metadata
    stim_count = {}
    selected_indices = []
    
    for idx, stim_name in enumerate(metadata['stimName']):
        if stim_name not in stim_count:
            stim_count[stim_name] = 0
        stim_count[stim_name] += 1
        
        if stim_count[stim_name] == n:
            selected_indices.append(idx)
    new_epochs = epochs[selected_indices]
    
    return new_epochs


def unique_class_epochs(epochs,n=1):
    """
    返回第n个的class的epoch
    """
    metadata = epochs.metadata
    class_count = {}
    selected_indices = []
    
    for idx, class_id in enumerate(metadata['class_id']):
        if class_id not in class_count:
            class_count[class_id] = 0
        class_count[class_id] += 1
        
        if class_count[class_id] == n:
            selected_indices.append(idx)
    new_epochs = epochs[selected_indices]
    
    return new_epochs

def kknc(data: np.ndarray, n: int or None = None):
    """
    Calculate the noise ceiling reported in the NSD paper (Allen et al., 2021)
    Arguments:
        data: np.ndarray
            Should be shape (ntargets, nrepetitions, nobservations)
            (channel num, 重复观看次数，不同图片的数目)
        n: int or None
            Number of trials averaged to calculate the noise ceiling. If None, n will be the number of repetitions.
    returns:
        nc: np.ndarray of shape (ntargets)
            Noise ceiling without considering trial averaging.
        ncav: np.ndarray of shape (ntargets)
            Noise ceiling considering all trials were averaged.
    """
    if not n:
        n = data.shape[-2]
    normalized = zscore(data, axis=-1)
    noisesd = np.sqrt(np.mean(np.var(normalized, axis=-2, ddof=1), axis=-1))
    sigsd = np.sqrt(np.clip(1 - noisesd ** 2, 0., None))
    ncsnr = sigsd / noisesd
    nc = 100 * ((ncsnr ** 2) / ((ncsnr ** 2) + (1 / n)))
    
    return nc

# def calculate_noise_ceiling(all_epochs,all_nc = {},n_repet = 20, n_imgs = 120):
#     n_time = len(all_epochs['01'].times)
#     for sub in co_sub:
#         print(sub)
#         epochs      = all_epochs[sub]
#         n_channels  = epochs.info['nchan']
#         epochs.load_data()
        
#         res_mat = np.empty([n_channels,n_repet,n_imgs,n_time])
#         for repet in range(n_repet):
#             epochs_curr = unique_stim_epochs(epochs, n=repet+1)
#             sort_order  = np.argsort(epochs_curr.metadata['stimName'])
#             epochs_curr = epochs_curr[sort_order]
#             epochs_curr = np.transpose(epochs_curr._data, (1,0,2))
#             res_mat[:,repet,:,:] = epochs_curr
            
#         nc = np.empty([n_channels,n_time])
#         for t in range(n_time):
#             dat = res_mat[:,:,:,t]
#             nc[:,t] = kknc(data = dat,n = n_repet)  
#         all_nc[sub] = nc
        
#     return all_nc

def calculate_noise_ceiling(all_epochs,all_nc = {},n_repet = 4, n_class = 1000):
    n_time = len(all_epochs['01'].times)
    for sub in all_epochs:
        print(sub)
        epochs      = all_epochs[sub]
        n_channels  = epochs.info['nchan']
        epochs.load_data()
        
        res_mat = np.empty([n_channels,n_repet,n_class,n_time])
        for repet in range(n_repet):
            epochs_curr = unique_class_epochs(epochs, n=repet+1)
            sort_order  = np.argsort(epochs_curr.metadata['class_id'])
            epochs_curr = epochs_curr[sort_order]
            epochs_curr = np.transpose(epochs_curr._data, (1,0,2))
            res_mat[:,repet,:,:] = epochs_curr
            
        nc = np.empty([n_channels,n_time])
        for t in range(n_time):
            dat = res_mat[:,:,:,t]
            nc[:,t] = kknc(data = dat,n = n_repet)  
        all_nc[sub] = nc
        
    return all_nc

def make_main_plot_class(all_epochs, all_nc):
    plt.close('all')
    fig = plt.figure(num=1, figsize=(12, 3))
    gs1 = gridspec.GridSpec(1, len(names), wspace=0.1)
    ctf_layout = mne.find_layout(all_epochs['01'].info)

    for i, n in enumerate(names):
        ax = fig.add_subplot(gs1[i])

        picks_epochs = {sub: np.where([s[2] == n for s in all_epochs[sub].ch_names])[0] for sub in all_epochs}
        picks = np.where([i[2] == n for i in ctf_layout.names])[0]

        for sub in all_epochs:
            times = all_epochs[sub].times * 1000
            mean_nc = np.mean(all_nc[sub][picks_epochs[sub], :], axis=0)
            ax.plot(times, mean_nc, color=colors(int(sub)), alpha=0.7, lw=2, label=f'P{int(sub)}')  

        all_means = np.array([np.mean(all_nc[sub][picks_epochs[sub], :], axis=0) for sub in all_epochs])
        overall_mean = np.mean(all_means, axis=0)
        ax.plot(times, overall_mean, color='black', lw=4, label='Mean')

        ax.set_ylim([0, 20])
        ax.set_xlim([times[0], times[-1]])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i == len(names) - 1:
            ax.legend(frameon=False, bbox_to_anchor=(1.6, 0.5), loc='center right')
        
        if i == 0:
            ax.set_ylabel('Explained variance (%)',fontsize = fontsize_other)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        ax.set_xlabel('time (ms)',fontsize = fontsize_other)
        ax.tick_params(axis='both', which='major', labelsize=fontsize_other)
        ax2 = ax.inset_axes([0.65, 0.65, 0.40, 0.40])
        ax2.plot(ctf_layout.pos[:, 0], ctf_layout.pos[:, 1], color='darkgrey', marker='.', linestyle='', markersize=2)
        ax2.plot(ctf_layout.pos[picks, 0], ctf_layout.pos[picks, 1], color='k', marker='.', linestyle='', markersize=2)
        ax2.axis('equal')
        ax2.axis('off')
        ax2.set_title(labs[i], y=0.8, fontsize=fontsize_other)
    
    # plt.tight_layout()
    fig.subplots_adjust(left=0.05,right=0.9, bottom=0.15, top=0.85)  
    fig.savefig(f'{fig_dir}/quality-noiseceiling_class.svg',dpi = 300)
#%%
def temp_meta(epo):
    meta = epo.metadata
    all_imgs = list(meta['image_id'])
    class_id = [i.split('_')[0] for i in all_imgs]
    meta['class_id'] = class_id
    epo.metadata = meta
    return epo

#%%
epo_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD/derivatives/epochs/task-ImageNet'
all_epo = all_epos(epo_root,subs = subs[:9])
for _,epo in all_epo.items():
    epo = temp_meta(epo)
all_nc = calculate_noise_ceiling(all_epo,n_repet = 4, n_class = 1000)
np.save('/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_codes/MEG/all_nc.npy', all_nc)
make_main_plot_class(all_epo, all_nc)


#%%
all_nc = np.load('/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_codes/MEG/all_nc.npy',allow_pickle=True).item()
make_main_plot_class(all_epo, all_nc)

#%%

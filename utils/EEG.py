#%%
import numexpr as ne
ne.set_num_threads(8)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt = '%m%d-%H:%M')

import os.path as op
import os
from pathlib import Path
from pprint import pprint
import json
from collections import defaultdict
import io
import contextlib
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore', category=np.ComplexWarning)


import mne_bids
from mne_bids import BIDSPath, read_raw_bids, get_entities_from_fname
import mne
from meegkit import dss
from meegkit.dss import dss0
from meegkit.utils import fold, unfold, tscov
from meegkit.asr import ASR
from pyprep.find_noisy_channels import NoisyChannels
from mne_icalabel import label_components
from scipy.io import loadmat
import math


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
from plotly.subplots import make_subplots
import os
import plotly.graph_objs as go
plt.rcParams['figure.max_open_warning'] = 60


mne.use_log_level(verbose='ERROR')
mne.cuda.init_cuda(verbose=True)


#%%
def tasks():
    return ['ImageNet']
    
def get_sub_list():
    return [f'{sub:02d}' for sub in range(1, 31) if f'{sub:02d}' not in missing_subs]    

def get_ses_dict():
    ses_d = {sub:[] for sub in get_sub_list()}
    for sub in ses_d:
        if int(sub) < 10:
            ses_d[sub] = [f'{ses:02d}' for ses in range(1, 5)]
        if int(sub) == 17:
            ses_d[sub] = ['02']
        else:
            ses_d[sub] = ['01','02']
    return ses_d

def get_run_dict():
    run_d = {}
    for sub in get_sub_list():
        run_d[sub] = {}
        for ses in get_ses_dict()[sub]:
            run_d[sub][ses] = [f'{run:02d}' for run in range(1, 9)]
    run_d['18']['01'] = ['03','04','05','06','07','08']
    del run_d['28']['01'][-1]
    return run_d


def get_subs():
    return ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
            '11', '12', '13', '14', '21', '24', '26', '27', '29', '30']


def events_plot(raw):
    # tri_raw = raw.copy().pick(['Trigger'])
    events, event_id = mne.events_from_annotations(raw)
    alltime = raw.times.shape[0] / raw.info['sfreq']

    interst_t = {}
    for i in event_id:
        tri = event_id[i]
        ntri = np.sum(events[:, 2] == tri)
        interst_t[i] = []
        t = events[events[:, 2] == tri, 0] / raw.info['sfreq']
        interst_t[i] = t
        pprint(f'Event {i} : {ntri}')

    intervals = {}
    for i in interst_t:
        t = interst_t[i]
        intervals[i] = np.diff(t)

    for i in intervals:
        avg_interval = np.mean(intervals[i])
        pprint(f'Average interval for Event {i}: {avg_interval} seconds')

    fig = make_subplots(rows=len(event_id), cols=1, shared_xaxes=True, subplot_titles=[f'avg_interval({i}) : {np.mean(intervals[i]):.2f}' for i in event_id])

    for idx, i in enumerate(event_id):
        tri = event_id[i]
        t = events[events[:, 2] == tri, 0] / raw.info['sfreq']

        fig.add_trace(go.Scatter(x=t, y=np.zeros_like(t) + idx, mode='markers', name=f'Event {i}', marker=dict(size=8)),
                      row=idx + 1, col=1)

        fig.update_yaxes(title_text=f'{i}', row=idx + 1, col=1)
        fig.update_xaxes(title_text='Time (s)', col=1)

    fig.update_layout(title='EEG timeline', showlegend=True, height=800, width=1000)

    fig.show()

def event_checkor(raw):
    ev,event_id = mne.events_from_annotations(raw)
    dsn_event_id = {'boundary': 4, '2':       1, '4':    2, '8':   3}

    if dsn_event_id != event_id:
        warnings.warn(f'{op.basename(raw.filenames[0])}: {event_id}')
        return False
    else:
        return True


def update_annot(raw):
    
    ev,ori_event_id    = mne.events_from_annotations(raw)
    if event_checkor(raw):
        new_event_id = {'begin':    4, 'stim_on': 1, 'resp': 2, 'end': 3}
    else:
        new_event_id = {'begin': ori_event_id.get('boundary',ori_event_id['700010']), 
                        'stim_on': ori_event_id['2'], 
                        'resp': ori_event_id['4'], 
                        'end': ori_event_id['8']}
        
        pprint(new_event_id)
    id_to_desc  = {v: k for k, v in new_event_id.items()}
    onsets = raw.times[ev[:, 0]]
    durations = np.repeat(1 / raw.info['sfreq'], len(onsets))
    descriptions = np.array([id_to_desc[code] for code in ev[:, 2]])

    new_annot = mne.Annotations(onset=onsets, duration=durations, 
                                description=descriptions, orig_time=raw.info['meas_date'])
    modified_raw = raw.copy()
    modified_raw.set_annotations(new_annot)
    
    moev, moevent_id = mne.events_from_annotations(modified_raw)
    
    return modified_raw
            

def baseline_correction(epochs):
    baselined_epochs = mne.baseline.rescale(data=epochs.get_data(),times=epochs.times,baseline=(None,0),mode='zscore',copy=False)
    epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin,event_id=epochs.event_id, metadata=epochs.metadata)
    return epochs


def concat_epo(epo_list):
    ref_head_t = epo_list[0].info['dev_head_t']
    for epo in epo_list:
        epo.info['dev_head_t'] = ref_head_t
    return mne.concatenate_epochs(epochs_list=epo_list, add_offset=True)


class EEGPreprocessing:
    def __init__(self, bids_path):
        self.process_info = {}
        self.extra_info   = {}
        self.bids_path = bids_path
        self.raw = read_raw_bids(self.bids_path)
        self.raw.load_data()
        # self.rename_dict = {'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz',
        #                     'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz',
        #                     'CPZ': 'CPz', 'PZ': 'Pz', 'POZ': 'POz', 'OZ': 'Oz'}
        # self.montage = mne.channels.make_standard_montage('standard_1005')
        
        # if 'NaN' in self.raw.annotations.description:
        #     self.raw.annotations.delete(np.where(self.raw.annotations.description == 'NaN'))
        # self.raw = update_annot(self.raw)
        if np.sum(self.raw.annotations.description == 'stim_on') != 125:
            # raise ValueError(f'events unmatched--{op.basename(self.raw.filenames[0])}: {np.sum(self.raw.annotations.description == "stim_on")}')
            warnings.warn(f'events unmatched--{op.basename(self.raw.filenames[0])}: {np.sum(self.raw.annotations.description == "stim_on")}')
        self.process_info['runinfo'] = self.bids_path.basename
        
        
        logging.info(f'preprocessing - {self.process_info["runinfo"]}')
        
    # def apply_montage(self, plot=False):
    #     self.raw.set_channel_types({"HEO": "eog", "VEO": "eog"})
    #     self.raw.rename_channels(self.rename_dict)
    #     if 'CB1' in self.raw.ch_names:
    #         self.raw.drop_channels(['CB1', 'CB2']) 
    #     if 'EKG' in self.raw.ch_names:
    #         self.raw.set_channel_types({"EKG": "bio"})
    #     if 'EMG' in self.raw.ch_names:
    #         self.raw.set_channel_types({"EMG": "emg"})
            
    #     self.raw.set_montage(self.montage)
    #     if plot:
    #         self.raw.plot_sensors(sphere=(0, 0.02, 0, 0.09))
        
    #     self.extra_info['montage'] = 'standard_1005'
    #     self.process_info['montage'] = 'DONE'
    #     # pprint(self.process_info)

        
        
    def find_bad_channels(self, is_interpolate=True):
        raw_nd = self.raw.copy()
        nd = NoisyChannels(raw_nd, random_state=1337)
        nd.find_bad_by_correlation()
        nd.find_bad_by_deviation()
        nd.find_bad_by_ransac()
        
        bads = nd.get_bads()
        raw_nd.info['bads'].extend(bads)
        if is_interpolate:
            raw_nd.interpolate_bads(reset_bads=True)
            
        
        self.raw = raw_nd
        self.process_info['bad channels fixing'] = 'DONE'
        self.extra_info['bad channels fixing'] = nd.get_bads(as_dict=True)
        # pprint(self.process_info)
        

    def zapline_denoise(self, fline = 50):
        data = self.raw.get_data().T
        data = np.expand_dims(data, axis=2)  # 现在形状为 (nsample, nchan, ntrial(1))
        sfreq = self.raw.info['sfreq']
        with contextlib.redirect_stdout(io.StringIO()): # mute for simplicity
            out, _ = dss.dss_line_iter(data, fline, sfreq, nfft=400)
            cleaned_raw = mne.io.RawArray(out.T.squeeze(), self.raw.info)
            cleaned_raw.set_annotations(self.raw.annotations)
        
        self.raw = cleaned_raw 
        self.process_info['zapline denoise'] = 'DONE'
        self.extra_info['zapline denoise'] = f'fline - {fline}'
        # pprint(self.process_info)

    def robust_reference(self):
        raw = self.raw.copy()
        params = {
            'ref_chs': 'eeg',
            'reref_chs': 'eeg',
        }
        reference = Reference(raw, params, ransac=True)
        reference.perform_reference()
        self.raw = reference.raw
                
    def rereference(self, ref='average'):
        refraw = self.raw.copy()
        refraw.set_eeg_reference(ref)
        self.raw = refraw
        
        if 're_reference bef ICA' in self.process_info:
            self.process_info['re_reference aft ICA'] = 'DONE'
            self.extra_info['re_reference aft ICA'] = ref
            
        self.process_info['re_reference bef ICA'] = 'DONE'
        self.extra_info['re_reference bef ICA'] = ref
        # pprint(self.process_info)        
    
    def ica_automark(self, save = True, ica_dir=None):
        
        
        raw_resmpl = self.raw.copy().pick_types(eeg=True).load_data()
        raw_resmpl.resample(200)
        raw_resmpl.filter(1, 40)
        

        ica = mne.preprocessing.ICA(method='fastica',random_state=97,n_components=40)
        ica.fit(raw_resmpl)
        
        ic_labels = label_components(raw_resmpl, ica, method="iclabel")
        exclude = [i for i,l in enumerate(ic_labels['labels']) if l not in ['brain','other']]
        ica.exclude = exclude
        
        if save:
            fn = self.process_info['runinfo']
            ica_dir = os.path.join(ica_dir, 'icafif')
            os.makedirs(ica_dir, exist_ok=True)
            ica.save(os.path.join(ica_dir, f'{fn}_ica.fif'), overwrite=True)

        self.ica = ica
        self.raw4plot = raw_resmpl
        self.process_info['ICA auto label'] = 'DONE'
        self.extra_info['ICA auto label'] = {i:l for i,l in enumerate(ic_labels['labels'])}
        self.extra_info['ICA auto score'] = {i:float(s) for i,s in enumerate(ic_labels['y_pred_proba'])}
        self.extra_info['ICA Nartifact'] = len(exclude)
        # pprint(self.process_info)
            
    def ica_plot(self, ica_dir=None):
        #set path for figs
        fn = self.process_info['runinfo']
        fig_dir = os.path.join(ica_dir, 'figs')
        ics_fig = os.path.join(fig_dir, 'ICs_all')
        proper_dir = os.path.join(fig_dir, 'IC_properties')
        source_fig = os.path.join(fig_dir, 'IC_sources')
        proper_run_dir = os.path.join(proper_dir, fn)
        for dir in [fig_dir, ics_fig, proper_dir, source_fig, proper_run_dir]:
            os.makedirs(dir, exist_ok=True)

        raw_ica         = self.ica
        raw_resmpl      = self.raw4plot
        labels          = self.extra_info['ICA auto label']
        scores          = self.extra_info['ICA auto score']
        
        # ICs all plot
        # n_components    = ica.n_components_
        # n_rows          = 5
        # n_cols          = 4
        # plt.ioff()
        # fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 20))
        # for i in range(n_components):
        #     row = i // n_cols
        #     col = i % n_cols
        #     ax = axes[row, col]
        #     raw_ica.plot_components(picks=i, axes=ax, sphere=(0, 0.02, 0, 0.09), show=False)
        #     ax.set_title(f'{i}-{labels[i]}', fontsize=40)
        # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.6)
        # plt.savefig(os.path.join(ics_fig, f'{fn}.png'))
        # plt.close(fig)
        
        
        n_components    = raw_ica.n_components_
        n_cols = 5
        n_rows = 8
        plt.ioff()
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
        for i in range(n_components):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_components > 1 else axes  # 当只有一个组件时，axes不是二维数组
            raw_ica.plot_components(picks=i, axes=ax, sphere=(0, 0.02, 0, 0.09), show=False)
            ax.set_title(f'{i} - {labels[i]}({scores[i]:.2f})', fontsize=40)

        # 移除多余的子图
        # for i in range(n_components, n_rows * n_cols):
        #     fig.delaxes(axes.flatten()[i])

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.6)
        plt.savefig(os.path.join(ics_fig, f'{fn}.png'))
        plt.close(fig)
        
        # every IC's properties plot
        for comp in range(n_components):
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            with contextlib.redirect_stdout(io.StringIO()):
                raw_ica.plot_properties(raw_resmpl, picks=comp, topomap_args = {'sphere' : (0, 0.02, 0, 0.09)}, show=False, verbose=False)
            plt.savefig(os.path.join(proper_run_dir, f'{comp}.png'))
            plt.close(fig)
        
        # ICA source plot
        raw_ica.plot_sources(raw_resmpl, picks = [i for i in range(0,20)], show=False)
        plt.savefig(os.path.join(source_fig, f'{fn}-0.png'))
        raw_ica.plot_sources(raw_resmpl, picks = [i for i in range(20,40)], show=False)
        plt.savefig(os.path.join(source_fig, f'{fn}-1.png'))
        plt.close('all') 
        plt.ion()  

        self.process_info['ICA plot'] = 'DONE'
        # pprint(self.process_info)

    def ica_reconst(self,exclude):
        
        raw = self.raw.copy()
        raw.load_data()
        raw.resample(200)
        raw.filter(0.1, 40)
        
        if exclude:
            self.ica.exclude = exclude
        
        self.ica.apply(raw)
        self.raw = raw        
        self.process_info['ICA reconst'] = 'DONE'
        self.extra_info['ICA reconst'] = 'bandfilter: 0.1-40\nsplfreq: 200'
        # pprint(self.process_info)
        
    
    def mypipeline(self,ica_dir,exclude):
        """
        Input: mne.raw
        
        1. set montage 1005
        2. bad channel detection (pyprep)
        3. denoise-line noise
        4. denoise-ASR
        4. ICA
        5. re-referencing
        
        Out put: mne.raw
        """
        # self.apply_montage()
        self.zapline_denoise()
        self.find_bad_channels()
        
        self.rereference()#like eeglab, re-reference before ICA
        self.ica_automark(ica_dir=ica_dir)
        self.ica_plot(ica_dir=ica_dir)
        self.ica_reconst(exclude=exclude)
        self.rereference() #re-reference after ICA also
        
        annot = self.raw.annotations
        sum_annot = {an: int(np.sum(annot.description == an)) for an in np.unique(annot.description)}
        self.extra_info['annot'] = sum_annot
                
        return self.raw


class EEGEpoching:
    def __init__(self, sub, rawroot, behavdir):
        self.process_info = {}
        self.extra_info   = {}
        self.rawps = sorted([f'{rawroot}/{file}' for file in os.listdir(rawroot) if file.startswith(f'sub{sub}') and file.endswith('.fif')])
        self.bhvps = sorted([f'{behavdir}/{op.basename(file).split(".")[0]}.mat' for file in self.rawps])
        self.basicinfo = {'sub': sub}
        
        self.process_info['runsinfo']   = f'{sub} - {len(self.rawps)} runs'
        logging.info(f'epoching - sub{sub}')

    def load_data(self):
        self.raws = [mne.io.read_raw_fif(raw) for raw in self.rawps]
        self.bhvs = [loadmat(bhv) for bhv in self.bhvps]
    


    def epoching(self, metadta_path, epoch_window = (-0.1, 0.8)):
        stim_info = pd.read_csv(metadta_path, low_memory=False)
        sub_epo = []
        for i, (raw, bhv) in enumerate(zip(self.raws, self.bhvs)):
            # print(raw)
            # print(bhv)
            # raw = update_annot(raw)
            events, event_id = mne.events_from_annotations(raw)
            
            imgTrial = [bhv['runStim'][i][0][0].split('.')[0] for i in range(bhv['runStim'].size)]
            subTrial = []
            for img in imgTrial:
                sub_class = stim_info[stim_info['image_id'] == img]['sub_class'].values[0]
                subTrial.append(sub_class)
            supTrial = [stim_info[stim_info['image_id'] == i]['super_class'].values[0] for i in imgTrial]
            ani_resp = [True if i == 1 else False for i in bhv['trial'][:,4]]
            ani_labe = [True if i == 1 else False for i in bhv['animate_label'][0]]
            judge_acur = ani_resp == ani_labe
            rt       = bhv['trial'][:,-1]

            metadata, new_events, new_event_id = mne.epochs.make_metadata(
                events      = events[events[:,2] == event_id['stim_on']],
                event_id    = {'stim_on': event_id['stim_on']},
                tmin        = epoch_window[0],
                tmax        = epoch_window[1],
                sfreq       = raw.info['sfreq']
                )
            stim_ind        = metadata.index[metadata['event_name'] == 'stim_on'].tolist()
            
            fn                                          = self.rawps[i]
            sub                                         = fn.split('/')[-1].split('_')[0][3:]
            ses                                         = fn.split('/')[-1].split('_')[1][3:5]
            run                                         = fn.split('/')[-1].split('_')[2][3:5]
            
            metadata['task']                            = ['ImageNet'] * len(metadata)
            metadata['subject']                         = [sub] * len(metadata)
            metadata['session']                         = [f"ImageNet{ses}"] * len(metadata)
            metadata['run']                             = [run] * len(metadata)
            metadata.loc[stim_ind, 'image_id']          = imgTrial
            metadata.loc[stim_ind, 'sub_class']         = subTrial
            metadata.loc[stim_ind, 'super_class']       = supTrial
            metadata.loc[stim_ind, 'stim_is_animate']   = ani_labe
            metadata.loc[stim_ind, 'resp_is_animate']   = ani_resp
            metadata.loc[stim_ind, 'resp_is_right']     = judge_acur
            metadata.loc[stim_ind, 'RT']                = rt
            
            epochs = mne.Epochs(
                        raw         = raw,
                        events      = new_events,
                        event_id    = new_event_id,
                        tmin        = epoch_window[0],
                        tmax        = epoch_window[1],
                        metadata    = metadata,
                        # baseline    = (None, 0),
                        picks       = 'eeg'
                        )
            epochs_corrected = baseline_correction(epochs)
            sub_epo.append(epochs_corrected)

        epoched  = concat_epo(sub_epo)
        metadata1 = epoched.metadata
        self.epoched = concat_epo(sub_epo)
        
        self.extra_info[sub] = epoched.metadata
        self.process_info[sub] = 'DONE'
        

        
        
    
            
#%%
def compare_gfp(evo0, evo1,pick):
    evo0 = evo0.pick([i for i,c in enumerate(evo0.ch_names) if c[0] == pick])
    evo1 = evo1.pick([i for i,c in enumerate(evo1.ch_names) if c[0] == pick])
    gfp0 = np.sqrt(np.mean(evo0.data ** 2, axis=0))
    gfp1 = np.sqrt(np.mean(evo1.data ** 2, axis=0))
    
    plt.figure()
    plt.plot(evo0.times, gfp0, label='evo0')
    plt.plot(evo1.times, gfp1, label='evo1')
    plt.xlabel('Time (s)')
    plt.ylabel('GFP')
    plt.title(pick)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sub = '16'
    ses = '01'
    run = '07'
    bids_dir = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD_MEEG/EEG/EEG-BIDS'
    preprocessor = EEGPreprocessing(subject=sub, task='ImageNet', session=f'ImageNet{ses}', run=run, bids_dir=bids_dir)
    preprocessor.mypipeline('', exclude=None)
#%%

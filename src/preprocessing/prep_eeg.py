#%%
import os
import os.path as op
import sys
import warnings
from pathlib import Path
from collections import defaultdict
import contextlib
import io
import shutil

# Numeric and Data Handling
import numpy as np
import pandas as pd
import math
from scipy.io import loadmat

# EEG Processing
import mne
from mne_bids import BIDSPath, read_raw_bids, get_entities_from_fname,find_matching_paths, make_report
from pyprep.find_noisy_channels import NoisyChannels
from meegkit import dss
from mne_icalabel import label_components  # 可区分：[‘brain’, ‘muscle artifact’, ‘eye blink’, ‘heart beat’, ‘line noise’, ‘channel noise’, ‘other’].

# Parallel Processing and printer
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
from tqdm_joblib import tqdm_joblib
from wasabi import msg
from pprint import pprint

# Plotting Libraries
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Matplotlib Configurations
plt.rcParams['figure.max_open_warning'] = 1000

# Warnings and Logging Configurations
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')

# Initialize CUDA for MNE
mne.use_log_level(verbose='ERROR')
mne.cuda.init_cuda(verbose=True)

#%%
def loading_(start="Processing..."):
    end = f'{start} Done!'
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = pd.Timestamp.now().strftime('%m-%d %H:%M:%S')
            try:
                with msg.loading(f'  {start} {start_time}'):
                    result = func(*args, **kwargs)
                end_time = pd.Timestamp.now().strftime('%m-%d %H:%M:%S')
                msg.good(f'{end} {end_time}')
                sys.stdout.flush()
                return result
            except Exception as e:
                msg.fail(f'Error in function "{func.__name__}": {str(e)}')
                raise
        return wrapper
    return decorator

class Preprocessing():
    def __init__(self, 
                 bids:BIDSPath, 
                 result_dir:str
                 ) -> None:
        runinfo = bids.basename.split('.')[0]
        msg.divider(f"prep - {runinfo}")
        with msg.loading(f'  Loading raw data'):
            
            # write bids info and load raw data
            self.bids = bids
            self.runinfo = runinfo
            self.raw = read_raw_bids(self.bids)
            self.raw.load_data()
            
            # define result dir, contains ica process file and cleaned data
            self.result_dir = result_dir
            self.ica_dir = f'{result_dir}/ica'
            self.clean_dir = f'{result_dir}/cleaned_raw'
            os.makedirs(self.ica_dir, exist_ok=True)
            os.makedirs(self.clean_dir, exist_ok=True)
            
        msg.good(f"Data Loaded!")

        
        #sidecars for process info and extra info
        self.process_info = {}
        self.extra_info   = {}
        self.process_info['runinfo'] = runinfo
        msg.text(f'for process details, check "process_info" and "extra_info" attributes')
        
    
    @loading_('Bad channels fixing...')
    def bad_channels_fixing(self, 
                            is_interpolate:bool=True
                            )->None:
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
    
    @loading_('Zapline denoising...')
    def zapline_denoise(self, 
                        fline:float = 50
                        )->None:
        data = self.raw.get_data().T
        data = np.expand_dims(data, axis=2)  # the shape now: (nsample, nchan, ntrial(1))
        sfreq = self.raw.info['sfreq']
        with contextlib.redirect_stdout(io.StringIO()): # mute for simplicity
            out, _ = dss.dss_line_iter(data, fline, sfreq, nfft=400)
            cleaned_raw = mne.io.RawArray(out.T.squeeze(), self.raw.info)
            cleaned_raw.set_annotations(self.raw.annotations)
        
        self.raw = cleaned_raw 
        self.process_info['zapline denoise'] = 'DONE'
        self.extra_info['zapline denoise'] = f'fline - {fline}'
    
    @loading_('Re-refrencing...')
    def rereference(self, 
                    ref:str='average'
                    )->None:
        refraw = self.raw.copy()
        refraw.set_eeg_reference(ref)
        self.raw = refraw
        
        if 're_reference bef ICA' in self.process_info:
            self.process_info['re_reference aft ICA'] = 'DONE'
            self.extra_info['re_reference aft ICA'] = ref
            
        self.process_info['re_reference bef ICA'] = 'DONE'
        self.extra_info['re_reference bef ICA'] = ref
    
    @loading_('ICA and ICs automarking...')
    def ica_automark(self, 
                     n_components=40, 
                     lfreq=1, 
                     hfreq=100, 
                     splr = 250, 
                     save = True
                     )->None:
        
        ica_dir=self.ica_dir
        raw_resmpl = self.raw.copy().pick_types(eeg=True).load_data()
        raw_resmpl.resample(splr)
        raw_resmpl.filter(lfreq, hfreq)

        ica = mne.preprocessing.ICA(method='infomax',random_state=97,n_components=n_components,max_iter='auto')
        ica.fit(raw_resmpl)
        ic_labels = label_components(raw_resmpl, ica, method="iclabel")
        exclude = [i for i,l in enumerate(ic_labels['labels']) if l not in ['brain','other']]
        ica.exclude = exclude
        
        if save:
            fn = self.process_info['runinfo']
            icafif_dir = os.path.join(ica_dir, 'icafif')
            os.makedirs(icafif_dir, exist_ok=True)
            ica.save(os.path.join(icafif_dir, f'{fn}_ica.fif'), overwrite=True)

        self.ica = ica
        self.raw4plot = raw_resmpl
        self.process_info['ICA'] = 'DONE'
        self.extra_info['ICA'] = {}
        self.extra_info['ICA']['bandfilter'] = f'{lfreq}-{hfreq}'
        self.extra_info['ICA']['sample rate'] = splr 

        self.process_info['ICA auto label'] = 'DONE'
        self.extra_info['ICA auto label'] = {i:l for i,l in enumerate(ic_labels['labels'])}
        self.extra_info['ICA auto score'] = {i:float(s) for i,s in enumerate(ic_labels['y_pred_proba'])}
        self.extra_info['ICA Nartifact'] = len(exclude)
        
    @loading_('ICs plotting...')
    def ica_plot(self):
        #set path for figs
        ica_dir = self.ica_dir
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

        n_components    = raw_ica.n_components_
        n_cols = 5
        n_rows = 8
        plt.ioff()
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
        for i in range(n_components):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_components > 1 else axes  
            raw_ica.plot_components(picks=i, axes=ax, sphere=(0, 0.02, 0, 0.09), show=False)
            if i in self.extra_info['ICA mark']:
                ax.set_title(f'{i} ', fontsize=50, color='red', fontweight='bold')
            else:
                ax.set_title(i, fontsize=40)

        
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
        
        if not op.exists(f'{fig_dir}/ICs_select_app.py'):
            shutil.copy(op.join(op.dirname(__file__), 'ICs_select_app.py'), f'{fig_dir}/')
            
        self.process_info['ICA plot'] = 'DONE'
    
    @loading_('ICs reconstraction...')
    def ica_reconst(self,
                    exclude=None, 
                    lfreq=0.1, 
                    hfreq=100, 
                    splr=250
                    )->None:
        
        raw = self.raw.copy()
        raw.load_data()
        raw.resample(splr)
        raw.filter(lfreq, hfreq)
        
        if exclude:
            self.ica.exclude = exclude
        
        self.ica.apply(raw)
        self.raw = raw        
        self.process_info['ICA reconst'] = 'DONE'
        self.extra_info['ICA reconst'] = {}
        self.extra_info['ICA reconst']['bandfilter'] = f'{lfreq}-{hfreq}'
        self.extra_info['ICA reconst']['sample rate'] = splr
        
    def my_pipeline(self, exclude=None)->None:
        """
        
        1. bad channel detection (pyprep)
        2. denoise-line noise(meegkit)
        3. re-referencing(MNE)
        3. ICA(MNE, MNE-ICAlabel)
        5. re-referencing(MNE)
        
        """
        self.bad_channels_fixing()
        self.zapline_denoise()
        
        self.rereference()#like eeglab, re-reference before ICA
        self.ica_automark()
        self.ica_plot()
        self.ica_reconst(exclude=exclude)
        self.rereference() #re-reference after ICA also
        
        annot = self.raw.annotations
        sum_annot = {an: int(np.sum(annot.description == an)) for an in np.unique(annot.description)}
        self.extra_info['annot'] = sum_annot
        self.raw.save(f'{self.clean_dir}/{self.runinfo}_clean.fif', overwrite=True)
        msg.info(f"Cleaned data saved in {self.clean_dir}")
        
    def manual0_pipeline(self):
        msg.info('Running before ICA manual exclude')
        self.bad_channels_fixing()
        self.zapline_denoise()
        self.rereference()#like eeglab, re-reference before ICA
        self.ica_automark()
        self.ica_plot()
        
        
    def manual1_pipeline(self, 
                         manual_exlude,
                         lfreq=0.1,
                         hfreq=100,
                         splr=250
                         ):
        msg.info('Running after ICA manual exclude')
        # preprocess raw for alignment
        self.bad_channels_fixing()
        self.zapline_denoise()
        self.rereference()
        
        # load ica and exclude artifact id
        icafif = f'{self.ica_dir}/icafif/{self.runinfo}_ica.fif'
        ica = mne.preprocessing.read_ica(icafif)
        ica.exclude = manual_exlude
        ica.save(icafif, overwrite=True)
        self.ica = ica
        self.extra_info['ICA mark'] = manual_exlude
        
        # plot prepare
        raw4plot = self.raw.copy().pick_types(eeg=True).load_data()
        raw4plot.resample(splr)
        raw4plot.filter(lfreq,hfreq)
        self.raw4plot = raw4plot
        self.ica_plot()
        
        # ica reconst and save cleaned data
        self.ica_reconst()
        self.rereference() #re-reference after ICA also
        self.raw.save(f'{self.clean_dir}/{self.runinfo}_clean.fif', overwrite=True)
        
        annot = self.raw.annotations
        sum_annot = {an: int(np.sum(annot.description == an)) for an in np.unique(annot.description)}
        self.extra_info['annot'] = sum_annot
        msg.info(f"Cleaned data saved in {self.clean_dir}")
        
#%%
if __name__ == "__main__":
    from info_extraction import ExtractInfo
    bids_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/NOD-EEG'
    design_events_id = {'begin': 1,'end': 2,'resp': 3,'stim_on': 4}
    n_stim = 125
    extension = 'set'
    
    info:ExtractInfo = ExtractInfo(bids_root, extension,'eeg')
    info.event_check(n_stim = n_stim, stim_id = 'stim_on', event_id = design_events_id)

    sub = '01'
    result_dir = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG/tests'
    bids = info.bids[sub][1]
    
    preprocessor:Preprocessing = Preprocessing(bids,result_dir)
    preprocessor.manual0_pipeline()
    preprocessor.manual1_pipeline([0,1,2])

#%%

#%%
# System Libraries
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

# MEG Processing
import mne
from mne_bids import BIDSPath, read_raw_bids, get_entities_from_fname,find_matching_paths, make_report
from meegkit import dss
from .prep_megnet import prepare_for_megnet
from .do_megnet import fPredictICA
from .info_extraction import ExtractInfo as ie

# Parallel Processing and printer
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
from tqdm_joblib import tqdm_joblib
from wasabi import msg

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
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize CUDA for MNE
mne.use_log_level(verbose='ERROR')
mne.cuda.init_cuda(verbose=True)

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

class Preprocessing:
    """
    """    
    def __init__(self, bids:BIDSPath, result_dir:str) -> None:
        runinfo = bids.basename.split('.')[0]
        msg.divider(f"prep - {runinfo}")
        msg.text(f'for process details, check "process_info" and "extra_info" attributes')
        with msg.loading(f'  Loading raw data'):
            self.bids = bids
            self.raw = read_raw_bids(self.bids)
            self.raw.load_data()
            self.result_dir = result_dir
            self.ica_dir = f'{result_dir}/ica'
            self.clean_dir = f'{result_dir}/cleaned_raw'
            self.runinfo = runinfo
            
            os.makedirs(self.ica_dir, exist_ok=True)
            os.makedirs(self.clean_dir, exist_ok=True)

        msg.good(f"Data Loaded!")
        
        #sidecars for process info and extra info
        self.process_info = {}
        self.extra_info   = {}
        self.process_info['runinfo'] = runinfo
        
    
    @loading_('Bad channels fixing...')
    def bad_channels_fixing(self)->None:
        """
        bad channels detection and interpolation for CTF MEG
        """
        raw = self.raw.copy()
        # if raw.compensation_grade != 0:
        #     raw.apply_gradient_compensation(0)
            
        auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
            raw=raw, 
            return_scores=True, 
            origin=(0,0,0.04), #(0,0,0,0.04) seems more reasonable for a adult head
            cross_talk=None, 
            calibration=None,
            )
        
        if not auto_noisy_chs and not auto_flat_chs:
            self.extra_info['bad_chs'] = 'None'
        else:
            self.extra_info['bad_chs'] = {'noisy': [str(ch) for ch in auto_noisy_chs], 'flat': [str(ch) for ch in auto_flat_chs]}
            self.extra_info['N_bad_chs'] = len(auto_noisy_chs+auto_flat_chs)
            
            raw.info['bads'].extend(auto_noisy_chs+auto_flat_chs)
            raw.interpolate_bads(reset_bads=True,origin=(0,0,0.04))
            # raw.apply_gradient_compensation(3)
            self.raw = raw
            
        self.process_info['bad channels'] = 'DONE'
    
    @loading_('Zapline denoising...')
    def zapline_denoise(self, fline = 50, nremove=60)->None:
        raw = self.raw.copy()
        
        raw = raw.pick(['mag'])
        data = raw.get_data().T
        data  = data[..., None]  # (nsample, nchan, ntrial(1))
        with contextlib.redirect_stdout(io.StringIO()): # mute for simplicity
            out, _ = dss.dss_line(data, 50, sfreq = raw.info['sfreq'], nremove=nremove, blocksize=1000, show=False)
        cleaned_data = out.T.squeeze()
        
        cleaned_raw = mne.io.RawArray(cleaned_data, raw.info)
        cleaned_raw.set_annotations(raw.annotations)
        self.raw = cleaned_raw
            
    @loading_('ICA process...')
    def ica_process(self,lfreq=1,hfreq=100,splr=250, n_components=40, save=False, megnet=True)->None:
        ica_dir = self.ica_dir        
        raw_resmpl = self.raw.copy().pick_types(meg=True, ref_meg=False).load_data()
        raw_resmpl.resample(splr)
        raw_resmpl.filter(lfreq, hfreq)
        
        ica = mne.preprocessing.ICA(method = 'infomax', random_state=97, n_components=n_components, max_iter='auto')
        ica.fit(raw_resmpl)
        
        fn = self.process_info['runinfo']
        if save:    
            icafif_dir = os.path.join(self.ica_dir, 'icafif')
            os.makedirs(icafif_dir, exist_ok=True)
            ica.save(os.path.join(icafif_dir, f'{fn}_ica.fif'), overwrite=True)
            
        if megnet:
            icamat_dir = f'{ica_dir}/icamat/{fn}'
            os.makedirs(icamat_dir, exist_ok=True)
            prepare_for_megnet(ica, raw_resmpl, icamat_dir)
        
        self.ica = ica
        self.raw4plot = raw_resmpl
        self.process_info['ICA'] = 'DONE'
        self.extra_info['ICA'] = {}
        self.extra_info['ICA']['bandfilter'] = f'{lfreq}-{hfreq}'
        self.extra_info['ICA']['sample rate'] = splr 
        self.icamat_dir = icamat_dir
    
    @loading_('ICA automark...')
    def ica_automark(self,model_path, n_jobs, save = True)->None:
        with contextlib.redirect_stdout(io.StringIO()): # mute for simplicity
            mark_result = fPredictICA(self.icamat_dir, model_path, Ncomp = self.ica.n_components_, n_jobs=n_jobs)
        
        exclude = list(np.where(mark_result != 0)[0])
        ica = self.ica # Directly writing to self.ica's exclude attribute won't work because ica is a copy
        ica.exlude = exclude
        if save:
            fn = self.process_info['runinfo']
            icafif_dir = os.path.join(self.ica_dir, 'icafif')
            os.makedirs(icafif_dir, exist_ok=True)
            ica.save(os.path.join(icafif_dir, f'{fn}_ica.fif'), overwrite=True)
            
        
        self.ica = ica  
        self.process_info['ICA automark'] = 'DONE'
        self.extra_info['ICA mark'] = exclude
           
    @loading_('ICs plotting...')
    def ica_plot(self)->None:
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
            raw_ica.plot_components(picks=i, axes=ax, show=False)
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
    def ica_reconst(self,exclude=None, lfreq=0.1, hfreq=100, splr=250)->None:
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
        
    def auto_pipeline(self, model_path, n_jobs)->None:
        """            
        1. bad channels fixing
        2. zapline denoising
        3. ICA process
        4. ICA automark
        5. ICA plot
        6. ICA reconstruction            
        """
        self.bad_channels_fixing()
        self.zapline_denoise()
        self.ica_process(save=False)
        self.ica_automark(model_path,n_jobs = n_jobs,save=True)
        self.ica_plot()
        self.ica_reconst()
        
        annot = self.raw.annotations
        sum_annot = {an: int(np.sum(annot.description == an)) for an in np.unique(annot.description)}
        self.extra_info['annot'] = sum_annot
        self.raw.save(f'{self.clean_dir}/{self.runinfo}_clean.fif', overwrite=True)
        msg.info(f"Cleaned data saved in {self.clean_dir}")
        
        
    def manual_pipeline0(self, model_path, n_jobs):
        # preprocess raw and ICA
        self.bad_channels_fixing()
        self.zapline_denoise()
        self.ica_process(save=False)
        self.ica_automark(model_path,n_jobs = n_jobs,save=True)
        self.ica_plot()
        
    def manual_pipeline1(self, manual_exlude,lfreq=0.1,hfreq=100,splr=250):
        # preprocess raw for alignment
        self.bad_channels_fixing()
        self.zapline_denoise()        
        # load ica and exclude artifact id
        icafif = f'{self.ica_dir}/icafif/{self.runinfo}_ica.fif'
        ica = mne.preprocessing.read_ica(icafif)
        ica.exclude = manual_exlude
        ica.save(icafif, overwrite=True) # save ica with manual exclude label
        self.ica = ica
        self.extra_info['ICA mark'] = manual_exlude
        
        # plot prepare
        raw4plot = self.raw.copy().pick_types(meg=True, ref_meg=False).load_data()
        raw4plot.resample(splr)
        raw4plot.filter(lfreq,hfreq)
        self.raw4plot = raw4plot
        self.ica_plot()
        
        # ica reconst and save cleaned data
        self.ica_reconst()
        self.raw.save(f'{self.clean_dir}/{self.runinfo}_clean.fif', overwrite=True)
        
        annot = self.raw.annotations
        sum_annot = {an: int(np.sum(annot.description == an)) for an in np.unique(annot.description)}
        self.extra_info['annot'] = sum_annot
        msg.info(f"Cleaned data saved in {self.clean_dir}")
#%%
if __name__ == '__main__':
    
    bids_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/NOD-MEG'
    output_dir = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_upload/NOD-MEG/derivatives'
    info = ie.ExtractInfo(bids_root, extension = 'fif', datatype = 'meg')
    bids = info.bids['02'][0]
    preprocessor = Preprocessing(bids, output_dir)
    preprocessor.manual0_pipeline('../../models/megnet_enigma.keras',n_jobs=8)
    preprocessor.manual1_pipeline([1,2,3,4])
    
#%%

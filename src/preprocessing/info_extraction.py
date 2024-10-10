#%%
import warnings
from pathlib import Path
from collections import defaultdict

# Numeric and Data Handling
import pandas as pd
import numpy as np

# Parallel Processing for Speed
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from wasabi import msg

# For BIDS Data Handling
import mne
from mne_bids import read_raw_bids, find_matching_paths, make_report

# Warnings and Logging Configurations
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')

# Initialize CUDA for MNE
mne.use_log_level(verbose='ERROR')

#%%
class ExtractInfo:
    """
    Extracts and processes information from a BIDS-compliant M/EEG dataset.

    Parameters
    ----------
    bids_root : str
        The root path to the BIDS-compliant dataset containing EEG/MEG data.
    extension : str
        The file extension of the EEG/MEG data files (e.g., 'set', 'fif').
    datatype : str
        The data type need to be extracted
    emptyroom : bool
        Whether to include the emptyroom data in the dataset.

    Attributes
    ----------
    bids_root : str
        The path to the root directory of the BIDS dataset.
    subs : list
        Subject identifiers extracted from the BIDS dataset.
    bids : dict
        A dictionary where the keys are subject identifiers and the values are lists of corresponding data file paths for each subject.
    
    Methods
    -------
    event_check(n_stim: int, stim_id: str, event_id: dict,n_jobs:int = -1) -> None
        Checks the event information in the dataset to ensure that the number of events matches the expected number of stimuli.
    """

    def __init__(self, bids_root:str, extension:str, datatype:str, emptyroom:bool = False)->None:            
        with msg.loading(f'  Initializing {bids_root.split("/")[-1]}...'):
            participants_file = Path(f'{bids_root}/participants.tsv')
            if not participants_file.exists():
                raise FileNotFoundError(f"BIDS data {participants_file} does not exist. Please check the BIDS path.")
            subinfo = pd.read_csv(participants_file, sep='\t')
            sub_id = sorted([s.split('-')[1] for s in subinfo['participant_id'].tolist()])
            sub_bids = {sub: find_matching_paths(bids_root, datatypes=datatype, subjects=sub, extensions=extension) for sub in sub_id}
            
            if 'emptyroom' in sub_bids and emptyroom is False:
                sub_bids.pop('emptyroom')
                sub_id.remove('emptyroom')
            
            self.bids_root = bids_root
            self.subs = sub_id
            self.bids = sub_bids
            self.report = make_report(self.bids_root)
            self.extension = extension
            
        msg.good(f"Initialized")
        
        msg.divider(f"Dataset info")
        print(f"Dataset root: {self.bids_root}")
        print(f"Data detection used extension: .{self.extension}, and data type: {datatype}")
        print(f"{len(self.subs)} subjects detected")
        print(f"{sum(len(runs) for runs in self.bids.values())} runs in all")
        
        print(f"\033[3mMNE-BIDS report as follows:\033[0m")
        msg.info(f'{self.report}')
        
    def event_check(self, n_stim: int, stim_id: str, event_id: dict,n_jobs:int = -1) -> None:
        def process_run(sub, run, event_id, n_stim, stim_id):
            raw = read_raw_bids(run)
            ev, ev_id = mne.events_from_annotations(raw)
            bad_evid = None
            bad_evnum = None
            if ev_id != event_id:
                bad_evid = run
            if np.sum(ev[:, 2] == event_id[stim_id]) != n_stim:
                bad_evnum = run
            return sub, bad_evid, bad_evnum

        bad_evid = defaultdict(list)
        bad_evnum = defaultdict(list)
        subjects_runs = [(sub, run) for sub, runs in self.bids.items() for run in runs]

        with tqdm_joblib(desc="Events checking", total=len(subjects_runs),leave=False) as progress_bar:
            results = Parallel(n_jobs=8)(
                delayed(process_run)(sub, run, event_id, n_stim, stim_id) for sub, run in subjects_runs
            )
        for sub, bad_evid_run, bad_evnum_run in results:
            if bad_evid_run:
                bad_evid[sub].append(bad_evid_run.run)
            if bad_evnum_run:
                bad_evnum[sub].append(bad_evnum_run.run)
        data = []
        
        subjects = set(bad_evid) | set(bad_evnum)
        for sub in subjects:
            data.append({
                'subject': sub,
                'wrong_id_runs': bad_evid.get(sub, []),
                'wrong_stims_runs': bad_evnum.get(sub, [])
            })
        df = pd.DataFrame(data)
        
        if df.empty or (df['subject'] == 'emptyroom').any(): # pass emptyroom for it has no stim info
            msg.good("No bad-events run detected across all subjects.")
        else:
            msg.fail("Bad-events runs detected")
            self.wrong_events_runs = df
            raise ValueError("There are bad-events runs detected. Please check the 'wrong_events_runs' attribute for details.")

#%%
if __name__ == '__main__':
    datatype  = 'eeg'
    extension = 'set'

    bids_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/NOD-EEG'
    design_events_id = {'begin': 1,'end': 2,'resp': 3,'stim_on': 4}
    n_stim    = 125

    info:ExtractInfo = ExtractInfo(bids_root, extension, datatype)
    info.event_check(n_stim = n_stim, stim_id = 'stim_on', event_id = design_events_id)

#%%

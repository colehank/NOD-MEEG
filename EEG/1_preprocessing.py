#%%
import os.path as op
import os
import sys
sys.path.append(op.abspath('..'))
import utils.EEG as nod
import json
import mne
import numpy as np
import pandas as pd
import logging
from pprint import pprint
import mne_bids
from mne_bids import read_raw_bids, BIDSPath


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt = '%m%d-%H:%M')
#%%
data_root           = '../../NOD-MEEG_data'
EEG_raw             = f'{data_root}/raw-BIDS/NOD-EEG'
behavdir            = f'{data_root}/behavior/EEG'
stimdir             = f'{data_root}/stimulus'

deri_dir            = f'{data_root}/derivatives'
preprocessed_dir    = f'{deri_dir}/preprocessed'
prep_raw            = f'{preprocessed_dir}/raw'
prep_epo            = f'{preprocessed_dir}/epo'
temp_dir            = f'{deri_dir}/temp'
for p in [deri_dir,preprocessed_dir,prep_raw, prep_epo, temp_dir]:
    os.makedirs(p, exist_ok=True)

#%%
subs = nod.get_subs()
process_infos = {}
extra_infos = {}
for sub in subs:
    sub_bids = mne_bids.find_matching_paths(EEG_raw,subjects=sub,extensions='set')
    raws = []
    for bids in sub_bids:
            ses = bids.session
            run = bids.run
            preprocessor        = nod.EEGPreprocessing(bids)
            preprocessed_raw    = preprocessor.mypipeline(ica_dir=f'{temp_dir}/ICA',exclude=None)
            process_infos[f'sub-{sub}_ses-{ses}_run-{run}'] = preprocessor.process_info
            extra_infos[f'sub-{sub}_ses-{ses}_run-{run}']   = preprocessor.extra_info
            
            with open(f'{prep_raw}/pre_processinfo.json', 'w') as f:
                json.dump(process_infos, f, indent=4)
            with open(f'{prep_raw}/pre_extrainfo.json', 'w') as f:
                json.dump(extra_infos, f, indent=4)
            preprocessed_raw.save(f'{prep_raw}/sub-{sub}_ses-{ses}_run-{run}.fif',overwrite=True)
            raws.append(preprocessed_raw)
            
            del preprocessed_raw, preprocessor
            pprint(process_infos[f'sub-{sub}_ses-{ses}_run-{run}'])
            
    epochor = nod.EEGEpoching(sub,rawroot = prep_raw,behavdir = behavdir)
    epochor.load_data()
    epochor.epoching(metadata_path = f'{stimdir}/metaStim.csv')
    epochor.epoched.save(f'{prep_epo}/sub-{sub}_epo.fif',overwrite=True)
    
    del raws, epochor    

            
#%%


#%%

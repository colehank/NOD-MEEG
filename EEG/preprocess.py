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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt = '%m%d-%H:%M')
#%%
EEG_root            = '..'
bids_dir            = f'{EEG_root}/EEG-BIDS'
temp_dir            = f'{EEG_root}/temp'
preprocess_dir      = f'{temp_dir}/preprocessed'
os.makedirs(preprocess_dir, exist_ok=True)

run_dict            = nod.get_run_dict()
process_infos       = {}
extra_infos         = {}
#%%
from mne_bids import read_raw_bids, BIDSPath

for sub in run_dict:
    # if int(sub) < 25:
        # continue
    for ses in run_dict[sub]:
        for run in run_dict[sub][ses]:
            if sub == '16' and ses == '02' and run == '01': #event missing,121 only
                continue
            if sub == '16' and ses == '01' and run == '07': #event missing,120 only
                continue
            if sub == '22' and ses == '02' and run == '01': #event missing,122 only
                continue
            if sub == '25' and ses == '01' and run == '06': #unreadable
                continue
            if sub == '25' and ses == '02' and run == '01':# unreadable
                continue
            
            preprocessor        = nod.EEGPreprocessing(subject=sub, task='ImageNet', session=f'ImageNet{ses}', run=run, bids_dir=bids_dir)
            preprocessed_raw    = preprocessor.mypipeline(ica_dir=f'{EEG_root}/temp/ICA',exclude=None)
            process_infos[f'sub{sub}_ses{ses}_run{run}'] = preprocessor.process_info
            extra_infos[f'sub{sub}_ses{ses}_run{run}']   = preprocessor.extra_info
            
            with open(f'{preprocess_dir}/pre_processinfo.json', 'w') as f:
                json.dump(process_infos, f, indent=4)
            with open(f'{preprocess_dir}/pre_extrainfo.json', 'w') as f:
                json.dump(extra_infos, f, indent=4)
            preprocessed_raw.save(f'{preprocess_dir}/sub{sub}_ses{ses}_run{run}.fif',overwrite=True)
            
            del preprocessed_raw, preprocessor
            pprint(process_infos[f'sub{sub}_ses{ses}_run{run}'])
            

            
#%%

#%%
import sys
import os.path as op
import os
import pickle
import json

# helper pakagee
sys.path.append(op.abspath('..'))
from src.preprocessing import info_extraction as ie
from src.preprocessing import prep_eeg as prep
from wasabi import msg

#%%
RAW_ROOT        = '../../NOD-MEEG_upload/NOD-EEG'
DESIGN_EVENT    =  {'begin': 1,'end': 2,'resp': 3,'stim_on': 4}
STIM_ID         = 'stim_on'
N_STIM          = 125
EXTENSION       = 'set'
DATATYPE        = 'eeg'
MAUNAL_ICA_PATH = f'{RAW_ROOT}/derivatives/ica/figs/Artifact_ICs.json'
MANUALED        = op.exists(MAUNAL_ICA_PATH)
msg.good(f'manual selection status: {MANUALED}') if MANUALED else msg.warn(f'manual selection status: {MANUALED}')
#%%
output_dir      = f"{RAW_ROOT}/derivatives"
os.makedirs(output_dir, exist_ok=True)
info            = ie.ExtractInfo(RAW_ROOT, EXTENSION, DATATYPE) # Initialize the whole dataset
info.event_check(n_stim = N_STIM, event_id = DESIGN_EVENT, stim_id = STIM_ID) # Check whether the events are correct as designed
#%%
# sidecar files
extra_info0      = {}
process_info0    = {}
extra_info1      = {}
process_info1    = {}

if not MANUALED:
    # Preprocess EEG before manual ICs selection
    for sub, bidss in info.bids.items(): 
        for bids in bidss:
            preprocessor = prep.Preprocessing(bids, output_dir)
            preprocessor.manual0_pipeline()
            
            extra_info0[preprocessor.runinfo]   = preprocessor.extra_info
            process_info0[preprocessor.runinfo] = preprocessor.process_info
    
    with open(f"{output_dir}/Preprocess_ExtraInfo0.pkl", 'wb') as f:
        pickle.dump(extra_info0, f)
    with open(f"{output_dir}/Preprocess_Info0.pkl", 'wb') as f:
        pickle.dump(process_info0, f)
        
else:
    # Preprocess EEG after manual ICs selection
    artifacts = json.load(open(MAUNAL_ICA_PATH))
    for sub, bidss in info.bids.items(): 
        for bids in bidss:
            preprocessor = prep.Preprocessing(bids, output_dir)
            runinfo = preprocessor.runinfo
            preprocessor.manual1_pipeline(artifacts[runinfo])
            
            extra_info1[preprocessor.runinfo]   = preprocessor.extra_info
            process_info1[preprocessor.runinfo] = preprocessor.process_info
    
    with open(f"{output_dir}/Preprocess_ExtraInfo1.pkl", 'wb') as f:
        pickle.dump(extra_info1, f)
    with open(f"{output_dir}/Preprocess_Info1.pkl", 'wb') as f:
        pickle.dump(process_info1, f)
            
#%%

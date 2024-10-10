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

#%%
RAW_ROOT        = '../../NOD-MEEG_upload/NOD-EEG'
DESIGN_EVENT    =  {'begin': 1,'end': 2,'resp': 3,'stim_on': 4}
STIM_ID         = 'stim_on'
N_STIM          = 125
EXTENSION       = 'set'
DATATYPE        = 'eeg'
MANUALED        = op.exists(f'{RAW_ROOT}/derivatives/ica/figs/artifact_ICs.json')
print(f'ICs manual selection sate: {MANUALED}')

#%%
output_dir      = f"{RAW_ROOT}/derivatives"
os.makedirs(output_dir, exist_ok=True)
info            = ie.ExtractInfo(RAW_ROOT, EXTENSION, DATATYPE) # Initialize the whole dataset
info.event_check(n_stim = N_STIM, event_id = DESIGN_EVENT, stim_id = STIM_ID) # Check whether the events are correct as designed

#%%
# sidecar files
# extra_info      = {}
# process_info    = {}

# for sub, bidss in info.bids.items(): 
#     for bids in bidss:
#         preprocessor = prep.Preprocessing(bids, output_dir)
#         preprocessor.my_pipeline()
#         extra_info[preprocessor.runinfo]   = preprocessor.extra_info
#         process_info[preprocessor.runinfo] = preprocessor.process_info

# with open(f"{output_dir}/Preprocess_ExtraInfo.pkl", 'wb') as f:
#     pickle.dump(extra_info, f)
# with open(f"{output_dir}/Preprocess_Info.pkl", 'wb') as f:
#     pickle.dump(process_info, f)

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
else:
    # Preprocess EEG after manual ICs selection
    artifacts = json.load(open(f'{RAW_ROOT}/derivatives/ica/figs/artifact_ICs.json'))
    for sub, bidss in info.bids.items(): 
        for bids in bidss:
            preprocessor = prep.Preprocessing(bids, output_dir)
            runinfo = preprocessor.runinfo
            preprocessor.manual1_pipeline(artifacts[runinfo])
            
            extra_info1[preprocessor.runinfo]   = preprocessor.extra_info
            process_info1[preprocessor.runinfo] = preprocessor.process_info
            
            
# Save the sidecar files
for extra_info, process_info, suffix in zip([extra_info0, extra_info1], [process_info0, process_info1], ['0', '1']):
    with open(f"{output_dir}/Preprocess_ExtraInfo{suffix}.pkl", 'wb') as f:
        pickle.dump(extra_info, f)
    with open(f"{output_dir}/Preprocess_Info{suffix}.pkl", 'wb') as f:
        pickle.dump(process_info, f)
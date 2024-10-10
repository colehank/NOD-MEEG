#%%
import sys
import os.path as op
import os
import pickle
import json
from wasabi import msg

# helper pakage
sys.path.append(op.abspath('..'))
from src.preprocessing import info_extraction as ie
from src.preprocessing import prep_meg as prep

#%%
RAW_ROOT        = '../../NOD-MEEG_upload/NOD-MEG'
DESIGN_EVENT    =  {'begin': 1,'end': 2,'resp': 3,'stim_on': 4}
STIM_ID         = 'stim_on'
N_STIM          = 200
EXTENSION       = 'fif'
DATATYPE        = 'meg'
MEGNET_MODEL    = '../models/megnet_enigma.keras'
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
    # Preprocess MEG before manual ICs selection
    for sub, bidss in info.bids.items(): 
        for bids in bidss:
            preprocessor = prep.Preprocessing(bids, output_dir)
            preprocessor.manual_pipeline0(model_path = MEGNET_MODEL, n_jobs = 8)
            
            extra_info0[preprocessor.runinfo]   = preprocessor.extra_info
            process_info0[preprocessor.runinfo] = preprocessor.process_info
    with open(f"{output_dir}/Preprocess_ExtraInfo0.pkl", 'wb') as f:
        pickle.dump(extra_info0, f)
    with open(f"{output_dir}/Preprocess_Info0.pkl", 'wb') as f:
        pickle.dump(process_info0, f)
    
else:
    # Preprocess MEG after manual ICs selection
    artifacts = json.load(open(MAUNAL_ICA_PATH))
    for sub, bidss in info.bids.items(): 
        for bids in bidss:
            preprocessor = prep.Preprocessing(bids, output_dir)
            preprocessor.manual_pipeline1(artifacts[preprocessor.runinfo])
            
            extra_info1[preprocessor.runinfo]   = preprocessor.extra_info
            process_info1[preprocessor.runinfo] = preprocessor.process_info
    with open(f"{output_dir}/Preprocess_ExtraInfo1.pkl", 'wb') as f:
        pickle.dump(extra_info1, f)
    with open(f"{output_dir}/Preprocess_Info1.pkl", 'wb') as f:
        pickle.dump(process_info1, f)
            
#%% Parallel processing
# from tqdm_joblib import tqdm_joblib
# from joblib import Parallel, delayed


# def process_bid(bids, output_dir, MEGNET_MODEL, artifacts=None):
#     preprocessor = prep.Preprocessing(bids, output_dir)
#     if artifacts is not None:
#         runinfo = preprocessor.runinfo
#         preprocessor.manual1_pipeline(artifacts[runinfo])
#     else:
#         preprocessor.manual0_pipeline(model_path=MEGNET_MODEL, n_jobs=8)
#     return preprocessor.runinfo, preprocessor.extra_info, preprocessor.process_info

# all_bids = [bids for bidss in info.bids.values() for bids in bidss]

# # 侧边文件
# extra_info = {}
# process_info = {}

# if not MANUALED:
#     with tqdm_joblib(desc="Processing", total=len(all_bids)) as progress_bar:
#         results = Parallel(n_jobs=8)(
#             delayed(process_bid)(bids, output_dir, MEGNET_MODEL) for bids in all_bids
#         )
# else:
#     artifacts = json.load(open(f'{RAW_ROOT}/derivatives/ica/figs/artifact_ICs.json'))
#     with tqdm_joblib(desc="Processing", total=len(all_bids)) as progress_bar:
#         results = Parallel(n_jobs=8)(
#             delayed(process_bid)(bids, output_dir, MEGNET_MODEL, artifacts) for bids in all_bids
#         )
        
# for runinfo, extra, process in results:
#     extra_info[runinfo] = extra
#     process_info[runinfo] = process
#%%

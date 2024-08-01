#%%
import os.path as op
import sys
sys.path.append(op.abspath('..'))
import utils.MEG as nod
import mne
from mne_bids import find_matching_paths,read_raw_bids
import numpy as np
import json
from pprint import pprint
#%%
data_dir = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/NOD-MEG'
metaStim = op.join(op.dirname(data_dir),'stimulus','metaStim.csv')
evnt_dir = op.join(data_dir,'events')
deri_dir = op.join(data_dir,'derivatives')
prep_dir = op.join(deri_dir,'preprocessed')
praw_dir = op.join(prep_dir,'raw')
pepo_dir = op.join(prep_dir,'epochs')
ica_dir  = op.join(prep_dir,'ICA')

for d in [evnt_dir,deri_dir, prep_dir,praw_dir,pepo_dir,ica_dir]:
    os.makedirs(d,exist_ok = True)

bids = find_matching_paths(data_dir,extensions = 'fif')
subs = list(np.unique([b.subject for b in bids]))
for sub in subs:
    if sub not in [f'{s:02d}' for s in range(1,31)]:
        raise ValueError(f'No bids found for subject {sub}')
#%%
process_infos = {}
extra_infos = {}
for sub in subs[:2]:
    raws = []
    sub_bids = find_matching_paths(data_dir,subjects = sub,extensions = 'fif')
    for bids in sub_bids:
        ses = bids.session
        run = bids.run
        preprocessor = nod.MEGPreprocessing(bids)
        praw = preprocessor.mypipeline(ica_dir)
        praw.save(f'{praw_dir}/sub-{sub}_ses-{ses}_run-{run}_pre.fif',overwrite = True)
        praw.append(raws)
        
        process_infos[f'sub-{sub}_ses-{ses}_run-{run}'] = preprocessor.process_info
        extra_infos[f'sub-{sub}_ses-{ses}_run-{run}'] = preprocessor.extra_info
        with open(f'{praw_dir}/pre_processinfo.json','w') as f:
            json.dump(process_infos,f,indent = 4)
        with open(f'{praw_dir}/pre_extrainfo.json','w') as f:
            json.dump(extra_infos,f,indent = 4)
        
        del preprocessor,praw
        pprint(process_infos[f'sub-{sub}_ses-{ses}_run-{run}'])
    
    epochor = nod.MEGEpoching(sub,rawroot = praw_dir,behavdir = behavdir,metadata_path = metaStim)
    pepo,sevent = epochor.mypipeline()
    pepo.save(f'{pepo_dir}/sub-{sub}_epo.fif',overwrite = True)
    sevent.to_csv(f'{evnt_dir}/sub-{sub}_events.csv',index = False)
#%%

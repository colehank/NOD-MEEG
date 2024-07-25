#%%
import mne
from mne_bids import BIDSPath, read_raw_bids,write_raw_bids
import os.path as op
import os
import sys
sys.path.append(op.abspath('..'))
import utils.EEG as nod

missing_subs = ['15', '17','18', '19', '20', '23', '28']
databad_subs = ['16', '22', '25']
bad_subs = missing_subs + databad_subs

subs = [f'{sub:02d}' for sub in range(1, 31) if f'{sub:02d}' not in missing_subs]
sess = ['01', '02']
runs = [f'{run:02d}' for run in range(1, 9)]
ori_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_all/EEG/EEG-BIDS'
new_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/EEG-BIDS'
os.makedirs(new_root,exist_ok=True)
#%%
def bidsupdate(bids,new_root):
    sub = bids.subject
    ses = bids.session
    run = bids.run
    task = bids.task
    
    new_bids = BIDSPath(subject=sub, session=ses, run=run, task = task, 
                        root=new_root,datatype='eeg')
    
    init_raw = nod.EEGPreprocessing(bids)
    init_raw.apply_montage()
    raw = init_raw.raw
    
    events,event_id = mne.events_from_annotations(raw)
    
    
    write_raw_bids(init_raw.raw, new_bids,events = events, 
                        event_id = event_id,montage=raw.get_montage(),allow_preload=True,format='EEGLAB',overwrite=True)
    
    
    
task = 'ImageNet'
for sub in subs:
    if int(sub) < 10:
        continue
    sess = [f'{task}{ses:02d}' for ses in range(1, 5)] if int(sub) < 10 else [f'{task}{ses:02d}' for ses in range(1, 3)]
    for ses in sess:
        runs = [f'{run:02d}' for run in range(1, 9)]
        for run in runs:
            bids = BIDSPath(subject=sub, session=ses, task=task, run=run, root=ori_root)
            bidsupdate(bids,new_root)


#%%

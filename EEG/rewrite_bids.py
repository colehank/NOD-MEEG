#%%
import mne
from mne_bids import BIDSPath, read_raw_bids,write_raw_bids
import os.path as op
import os
import sys
sys.path.append(op.abspath('..'))
import utils.EEG as nod
import pandas as pd
from scipy.io import loadmat
import re
from datetime import datetime, timezone
import shutil

missing_subs = ['15', '17','18', '19', '20', '23', '28']
databad_subs = ['16', '22', '25']
bad_subs = missing_subs + databad_subs

subs = [f'{sub:02d}' for sub in range(1, 31) if f'{sub:02d}' not in bad_subs]
sess = ['01', '02']
runs = [f'{run:02d}' for run in range(1, 9)]
ori_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_all/EEG/EEG-BIDS'
new_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/EEG-BIDS'
os.makedirs(new_root,exist_ok=True)
submap = pd.read_csv('/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_all/helpers/submap.tsv', sep='\t')
bhv_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/behavior/EEG'
#%%
def extract_datetime_from_metadata(metadata):
    metadata_str = metadata.decode('utf-8')
    match = re.search(r'Created on: (\w{3} \w{3} {1,2}\d{1,2} \d{2}:\d{2}:\d{2} \d{4})', metadata_str)
    if match:
        date_str = match.group(1).strip()
        date_obj = datetime.strptime(date_str, '%a %b %d %H:%M:%S %Y')
        date_obj = date_obj.replace(tzinfo=timezone.utc)
        return date_obj
    return None
    
def bidsupdate(bids,new_root):
    sub = bids.subject
    ses = bids.session
    run = bids.run
    task = bids.task
    sex = 1 if submap[submap['subject_id'] == int(sub)]['sex'].values[0] == 'M' else 2
    age = submap[submap['subject_id'] == int(sub)]['age'].values[0]
    bhv = loadmat(f'{bhv_root}/sub-{sub}_ses-{ses}_run-{run}_eeg.mat')
    date = extract_datetime_from_metadata(bhv['__header__'])
    birthday = (date.year - int(age), date.month, date.day)
    new_bids = BIDSPath(subject=sub, session=ses, run=run, task = task, 
                        root=new_root,datatype='eeg')
    
    init_raw = nod.EEGPreprocessing(bids)
    init_raw.apply_montage()
    raw = init_raw.raw.copy()

    raw.info['line_freq'] = 50
    raw.info['subject_info'] = {
        'his_id':sub,
        'id':int(sub),
        'birthday':birthday,
        'sex':sex
    }
    raw.info['experimenter'] = 'Cibol'
    raw.set_meas_date(date)
    
    
    write_raw_bids(raw, new_bids,montage=raw.get_montage(),allow_preload=True,format='EEGLAB',overwrite=True)
    
    del raw,init_raw
    
    

task = 'ImageNet'

for sub in subs:
    sess = [f'{task}{ses:02d}' for ses in range(1, 5)] if int(sub) < 10 else [f'{task}{ses:02d}' for ses in range(1, 3)]
    for ses in sess:
        runs = [f'{run:02d}' for run in range(1, 9)]
        for run in runs:
            bids = BIDSPath(subject=sub, session=ses, task=task, run=run, root=ori_root)
            bidsupdate(bids, new_root)



#%% EEG behavior data
# rawbhv = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_all/EEG/raw_EEG/behaviordata'
# new_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/behavior/EEG'
# os.makedirs(new_root,exist_ok=True)

# task = 'ImageNet'

# for sub in subs:
#     ori_sub = f"{submap[submap['subject_id'] == int(sub)]['eeg_participant_id'].values[0]:02d}"
#     ori_sess = [f'{ses:02d}' for ses in range(1, 5)] if int(sub) < 10 else [f'{ses:02d}' for ses in range(1, 3)]
#     for ses in ori_sess:
#         runs = [f'{run:02d}' for run in range(1, 9)]
#         for run in runs:
#             print(f'{ori_sub} <-- {sub}')
#             ori_fp = op.join(rawbhv, f'sub{ori_sub}', f'sess{ses}', f'sub{ori_sub}_sess{ses}_run{run}.mat')
#             dst_fp = op.join(new_root, f'sub-{sub}_ses-{task}{ses}_run-{run}_eeg.mat')
#             shutil.copy(ori_fp, dst_fp)
# 
# #%% MEG behavior data
# rawbhv = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD/data/runfiles'
# new_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/behavior/MEG'
# os.makedirs(new_root,exist_ok=True)

# mat_files = []
# for root, dirs, files in os.walk(rawbhv):
#     for file in files:
#         if file.endswith(".mat"):
#             mat_files.append(os.path.join(root, file))

# for mat in mat_files:
#     sub = mat.split('/')[-1].split('_')[0][4:]
#     ses = mat.split('/')[-2].split('-')[1]
#     run = mat.split('/')[-1].split('_')[-1][3:5]
#     print(f'{sub} {ses} {run}')
#     dst_fp = op.join(new_root, f'sub-{sub}_ses-{ses}_run-{run}_meg.mat')
#     shutil.copy(mat, dst_fp)

#%%

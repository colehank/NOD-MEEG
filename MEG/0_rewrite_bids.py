#%%
import mne
from mne_bids import BIDSPath, read_raw_bids,write_raw_bids
import mne_bids
import os.path as op
import os
import sys
sys.path.append(op.abspath('..'))
import utils.MEG as nod
import pandas as pd
from scipy.io import loadmat
import re
from datetime import datetime, timezone
import shutil
import mne_bids
import json

subs = [f'{sub:02d}' for sub in range(1, 31)]
sess = {sub: [f'ImageNet{s:02d}' for s in range(1, 5)] if int(sub) < 10 else [f'ImageNet{s:02d}' for s in range(1, 3)] for sub in subs}
# for sub in sess:
#     if int(sub) < 10:
#         sess[sub].extend(['CoCo01','CoCo02'])\
runss = {sub: {ses: [f'{run:02d}' for run in range(1, 3)] for ses in sess[sub]} for sub in sess}
for sub, sessions in runss.items():
    if int(sub) < 10:
        for ses in ['ImageNet03', 'ImageNet04']:
            sessions[ses].extend([f'{r:02d}' for r in range(3, 9)])
    if int(sub) >= 10:
        runss[sub] = {'ImageNet01': [f'{r:02d}' for r in range(1, 6)]}

ori_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD/MEG-BIDS'
new_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/raw-BIDS/NOD-MEG'
os.makedirs(new_root,exist_ok=True)

bhv_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/behavior/MEG'
submap = pd.read_csv('/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_all/helpers/submap.tsv', sep='\t')

ctf_coord = json.load(open('/nfs/z1/userhome/zzl-zhangguohao/workingdir/NaturalObject/MEG/MEG-BIDS/sub-01/ses-ImageNet01/meg/sub-01_ses-ImageNet01_coordsystem.json'))
ctf_meg   = json.load(open('/nfs/z1/userhome/zzl-zhangguohao/workingdir/NaturalObject/MEG/MEG-BIDS/sub-01/ses-ImageNet01/meg/sub-01_ses-ImageNet01_task-ImageNet_run-01_meg.json'))
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
    bhv = loadmat(f'{bhv_root}/sub-{sub}_ses-{ses}_run-{run}_meg.mat')
    date = extract_datetime_from_metadata(bhv['__header__'])
    birthday = (date.year - int(age), date.month, date.day)
    new_bids = BIDSPath(subject=sub, session=ses, run=run, task = task, 
                        root=new_root,datatype='meg')
    
    raw = read_raw_bids(bids)
    raw.set_channel_types({'UPPT001':'stim'})
    raw.info['line_freq'] = 50
    raw.info['subject_info'] = {
        'his_id':sub,
        'id':int(sub),
        'birthday':birthday,
        'sex':sex
    }
    raw.info['experimenter'] = 'Cibol'
    raw.set_meas_date(date)
    
    write_raw_bids(raw, new_bids, allow_preload=True,format='FIF',overwrite=True)
    
    return new_bids
#%%
for sub,sess in runss.items():
    if int(sub) < 10:
        continue
    for ses,runs in sess.items():
        for run in runs:
            print(f'\rProcessing {sub} {ses} {run}',end = '', flush = True)
            bids = BIDSPath(subject=sub, session=ses, task='ImageNet', run=run, root=ori_root)
            
            new_bids = bidsupdate(bids,new_root)
            
            coordp = str(new_bids.find_matching_sidecar(suffix='_coordsystem',extension='.json'))
            megp  = str(new_bids.find_matching_sidecar(suffix='_meg',extension='.json'))
            coord = json.load(open(coordp, 'r', encoding='utf-8'))
            meg   = json.load(open(megp,  'r', encoding='utf-8'))
            
            for key in coord:
                if key in ['MEGCoordinateSystem', 'MEGCoordinateUnits', 'MEGCoordinateSystemDescription', 
                           'AnatomicalLandmarkCoordinateSystem', 'AnatomicalLandmarkCoordinateUnits']:
                    coord[key] = ctf_coord[key]
            for key in meg:
                if key == 'Manufacturer':
                    meg[key] = ctf_meg[key]

            with open(coordp, 'w', encoding='utf-8') as f:
                json.dump(coord, f, ensure_ascii=False, indent=4)
            with open(megp, 'w', encoding='utf-8') as f:
                json.dump(meg, f, ensure_ascii=False, indent=4)
                
            del coord, meg, coordp, megp, new_bids, bids
#%%
mri_dir = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_all/MRI/t1w'
all_mri = sorted([f"{mri_dir}/{f}" for f in os.listdir(mri_dir) if f.endswith('.nii.gz')])
all_side = sorted([f"{mri_dir}/{f}" for f in os.listdir(mri_dir) if f.endswith('.json')])

for sub in subs:
    print(f'\rProcessing {sub}',end = '', flush = True)
    bids_anat = BIDSPath(subject=sub, session = 'MRI', root=new_root, datatype='anat', suffix='T1w', extension='nii.gz')
    sidecar_anat = BIDSPath(subject=sub, session = 'MRI', root=new_root, datatype='anat', suffix='T1w', extension='json')
    mne_bids.write_anat(all_mri[int(sub)-1], bids_anat,overwrite=True,verbose=True)

#%%
mne_bids.make_dataset_description(path = new_root,name = 'NOD-MEG',dataset_type = 'raw', 
                                  authors = 'Guohao Zhang',overwrite=True)
print('\r',mne_bids.make_report(new_root))
#%%

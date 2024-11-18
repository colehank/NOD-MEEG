#%% Add face information to epochs
import pandas as pd
import os
import os.path as op
import mne

DATA_DIR = '../../NOD-MEEG_upload'
MEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-MEG', 'derivatives', 'preprocessed', 'epochs')
EEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-EEG', 'derivatives', 'preprocessed', 'epochs')

meg_epoch_files = {
    filename.split('-')[1][:2]: op.join(MEG_EPOCH_ROOT, filename)
    for filename in sorted(os.listdir(MEG_EPOCH_ROOT))
}
eeg_epoch_files = {
    filename.split('-')[1][:2]: op.join(EEG_EPOCH_ROOT, filename)
    for filename in sorted(os.listdir(EEG_EPOCH_ROOT))
}

#%%
faceinfo_p = f'{DATA_DIR}/NOD-stimulus/face.csv'
faceinfo_df = pd.read_csv(faceinfo_p)
faceinfo = faceinfo_df[['imgid', 'face_score', 'face_score_max', 'face_area', 'face_area_max']]

def append_faceinfo(epo_path: str, 
                    faceinfo: pd.DataFrame,
                    thresh: float = .8
                    ) -> mne.Epochs:
    epo = mne.read_epochs(epo_path)
    imgid = epo.metadata['image_id'].values
    faceinfo_dict = faceinfo.set_index('imgid').to_dict('index')
    
    face_score = []
    face_score_max = []
    face_area = []
    face_area_max = []
    stim_is_face = []
    
    for i in imgid:
        face_score.append(faceinfo_dict[i]['face_score'])
        face_score_max.append(faceinfo_dict[i]['face_score_max'])
        face_area.append(faceinfo_dict[i]['face_area'])
        face_area_max.append(faceinfo_dict[i]['face_area_max'])
        stim_is_face.append(
            faceinfo_dict[i]['face_score_max'] > thresh)
        
    epo.metadata['stim_is_face'] = stim_is_face
    epo.metadata['face_score'] = face_score
    epo.metadata['face_score_max'] = face_score_max
    epo.metadata['face_area'] = face_area
    epo.metadata['face_area_max'] = face_area_max
    
    return epo

for data in [meg_epoch_files, eeg_epoch_files]:
    for s, epo in data.items():
        print(f'\r{s}', end='', flush=True)
        # face_epo = append_faceinfo(epo, faceinfo)
        # face_epo.save(epo, overwrite=True)
#%%

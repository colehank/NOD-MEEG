#%%
import mne
from  scipy.io import loadmat
# import hcp
hcp_root = '/nfs/z1/HCP/HCPYA/MEG/preprocessed_data/resting'
sample_data_p = '/nfs/z1/HCP/HCPYA/MEG/preprocessed_data/resting/100307_MEG_Restin_preproc/100307/MEG/Restin/rmegpreproc'
raw = mne.io.read_raw_fieldtrip(sample_data_p, info=None)
# dat = loadmat(sample_data_p)

#%%

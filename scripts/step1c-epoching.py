#%%
import os
import os.path as op
import sys
sys.path.append(op.abspath('..'))
from src.epoching import Epoching
from src.epoching import InfoExtraction
from dataclasses import dataclass
import mne
import pandas as pd
from wasabi import msg
#%%
ROOT = '../../NOD-MEEG_upload'

MEG_ROOT, MEG_EVENT_ROOT, MEG_SAVE_ROOT = [
    f'{ROOT}/NOD-MEG/{path}' for path in [
        'derivatives/preprocessed/raw', 
        'events', 
        'derivatives/preprocessed/epochs']]
EEG_ROOT, EEG_EVENT_ROOT, EEG_SAVE_ROOT = [
    f'{ROOT}/NOD-EEG/{path}' for path in [
        'derivatives/preprocessed/raw', 
        'events',
        'derivatives/preprocessed/epochs'
        ]]

EVENT_ID = 'stim_on'
TMIN, TMAX = -0.1, 0.8
LFREQ, HFREQ = 0.1, 40
SFREQ = 200

#%%
class MakeEpochs:
    def __init__(
        self,
        roots: dict[str, str],
        event_roots: dict[str, str],
        event_id: str,
        tmin: float,
        tmax: float,
        lfreq: float,
        hfreq: float,
        sfreq: float,
        data_types: list[str],
        save_roots: dict
    ) -> None:
        """make epochs based on the NOD-MEG& EEG cleaned data.
        
        Parameters
        ----------
        roots : dict
            Paths to the cleaned data. Keys are data types(meg, eeg), values are the corresponding paths
        event_roots : dict
            Paths to the event data. Keys are data types(meg, eeg), values are the corresponding paths
        event_id : str
            The event ID to be used for epoching, shoud be stored in the mne.BaseRaw object
        tmin : float
            The start time of the epoch
        tmax : float
            The end time of the epoch
        lfreq : float
            The low pass filter frequency
        hfreq : float
            The high pass filter frequency
        sfreq : float
            The sampling frequency
        data_types : list
            the data types of the database, default is ['meg', 'eeg']
        save_roots : dict
            Paths to save the epochs. Keys are data types(meg, eeg), values are the corresponding paths
        """
        self.roots = roots  # {'meg': MEG_ROOT, 'eeg': EEG_ROOT}
        self.event_roots = event_roots  # {'meg': MEG_EVENT_ROOT, 'eeg': EEG_EVENT_ROOT}
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.lfreq = lfreq
        self.hfreq = hfreq
        self.sfreq = sfreq
        self.data_types = data_types  # ['meg', 'eeg']
        self.save_roots = save_root
        
        self.infos = {}
        for datatype in self.data_types:
            self.infos[datatype] = InfoExtraction(self.roots[datatype], self.event_roots[datatype])
    
    def make_sub(self, 
             sub: str,
             align_method:str) -> dict:
        """Create epochs for all data types for a subject
        
        Parameters
        ----------
        sub : str
            Subject ID
        align_method : str
            Method to align the data. 
            Options are 'info_with_data', 'info_only' and 'maxwell
        
        Returns
        -------
        dict
            Keys are data types, values are the corresponding epochs
        """
        epochs = {}
        for datatype in self.data_types:
            info = self.infos[datatype]
            try:
                sub_fps = info.get_sub_fp(sub)
            except KeyError:
                msg.warn(f"Subject {sub} not found in {datatype} data.")
                continue
            epochor = Epoching(
                event_csv=sub_fps['events'],
                raw_paths=sub_fps['rawps'],
                tmin=self.tmin,
                tmax=self.tmax,
                lfreq=self.lfreq,
                hfreq=self.hfreq,
                sfreq=self.sfreq,
                datatype=datatype,
                event_id=self.event_id
            )
            epoched = epochor.run(align_method = align_method)
            epochs[datatype] = epoched
            
        return epochs
    
    def run_all(self,
                align_method:str
                ) -> None:
        """Create and save epochs for all data types for all subjects"""
        subs = set()
        for info in self.infos.values():
            subs.update(info.subs)
        
        for sub in sorted(subs):
            epochs = self.make_sub(sub, align_method)
            for datatype, epoch in epochs.items():
                self._save(epoch, sub, self.save_roots[datatype], datatype)
    
    def _save(
        self, 
        epochs: mne.Epochs, 
        sub: str, 
        save_root: str, 
        datatype: str
        ) -> None:
        
        save_dir = os.path.join(save_root)
        os.makedirs(save_dir, exist_ok=True)
        epochs.save(f'{save_dir}/sub-{sub}_{datatype}_epo.fif', overwrite=True)
       
    def _repr_html_(self):
        root = (
            '<ul style="list-style-type:none; padding-left:0;">' +
            ''.join([f'<li>{k} : {v} </li>' for k, v in self.roots.items()]) +
            '</ul>'
        )
        event_root = (
            '<ul style="list-style-type:none; padding-left:0;">' +
            ''.join([f'<li>{k} : {v} </li>' for k, v in self.event_roots.items()]) +
            '</ul>'
        )
        nSub = (
            '<ul style="list-style-type:none; padding-left:0;">' +
            ''.join([f'<li>{k} : {len(v.subs)} </li>' for k, v in self.infos.items()]) +
            '</ul>'
        )
        save_root = (
            '<ul style="list-style-type:none; padding-left:0;">' +
            ''.join([f'<li>{k} : {v} </li>' for k, v in self.save_roots.items()]) +
            '</ul>'
        )
        to_show = {
            'rawRoot': root,
            'eventRoot': event_root,
            'nSub': nSub,
            'eventId': self.event_id,
            'timeMin': self.tmin,
            'timeMax': self.tmax,
            'lowFreq': self.lfreq,
            'highFreq': self.hfreq,
            'sampleFreq': self.sfreq,
            'dataTypes': self.data_types,
            'saveRoot': save_root
        }        
        to_show_df = pd.DataFrame(list(to_show.items()), columns=['', 'NOD_MEEG-Epochor'])        
        html_output = to_show_df.to_html(index=False, escape=False)  # escape=False to allow HTML in 'rawRoot'
        
        return html_output


#%%
roots = {'meg': MEG_ROOT, 'eeg': EEG_ROOT}
event_roots = {'meg': MEG_EVENT_ROOT, 'eeg': EEG_EVENT_ROOT}
data_types = ['meg', 'eeg']
save_root = {'meg': MEG_SAVE_ROOT, 'eeg': EEG_SAVE_ROOT}

epochor = MakeEpochs(
    roots=roots,
    event_roots=event_roots,
    event_id=EVENT_ID,
    tmin=TMIN,
    tmax=TMAX,
    lfreq=LFREQ,
    hfreq=HFREQ,
    sfreq=SFREQ,
    data_types=data_types,
    save_roots=save_root,
    )

epochor.run_all(align_method='info_with_data')
#%%

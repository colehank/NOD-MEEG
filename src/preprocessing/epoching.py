#%%
import mne
import os
from collections import defaultdict
from wasabi import msg
import pandas as pd
import re
import sys
mne.use_log_level(verbose='ERROR')
mne.cuda.init_cuda(verbose=True)


def loading_(start="Processing..."):
    end = f'{start} Done!'
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = pd.Timestamp.now().strftime('%m-%d %H:%M:%S')
            try:
                with msg.loading(f'  {start} {start_time}'):
                    result = func(*args, **kwargs)
                end_time = pd.Timestamp.now().strftime('%m-%d %H:%M:%S')
                msg.good(f'{end} {end_time}')
                sys.stdout.flush()
                return result
            except Exception as e:
                end_time = pd.Timestamp.now().strftime('%m-%d %H:%M:%S')
                msg.fail(f'Error in function "{func.__name__}": {str(e)}. Time: {end_time}')
                raise
        return wrapper
    return decorator


class InfoExtraction():
    def __init__(self, root:str, event_root:str)->None:
        all_events = {f.split('_')[0][-2:] : f'{event_root}/{f}' for f in sorted(os.listdir(event_root)) if f.endswith('.csv')}
        all_paths  = defaultdict(list)
        for p in sorted(os.listdir(root)):
            sub = p.split('_')[0][-2:]
            if p.endswith('.fif'):
                all_paths[sub].append(f"{root}/{p}") 
        
        self.raw_paths = all_paths
        self.event_paths = all_events
        self.tmin = tmin
        self.tmax = tmax
        self.subs = sorted(all_paths.keys())
        
        msg.text('all subject consistent in events and raws') if all([sub in all_events for sub in all_paths]) else msg.fail('not all subject consistent in events and raws')
    
    def sub_info(self, sub:str)->dict:
        return {'rawps':sorted(self.raw_paths[sub]), 'events':self.event_paths[sub]}
    
    
    
class Epoching():
    def __init__(self, event_csv:str, raw_paths:list, tmin:float, tmax:float, sfreq:float, lfreq:float, hfreq:float, datatype:str)->None:
        self.sub = re.search(r'sub-(\d+)_events\.csv', event_csv).group(1)
        msg.divider(f'Epoch-sub{self.sub}')
        self.event_path = event_csv
        self.raw_paths = raw_paths
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.lfreq = lfreq
        self.hfreq = hfreq
        self.datatype = datatype
        self.event_id = event_id
        
    
    @loading_('loading data & resample/filter if necessary')
    def load_data(self)->None:
        self.raws = [mne.io.read_raw_fif(rawp) for rawp in self.raw_paths]
        self.raws_info = sorted([rawp.split('/')[-1].split('.')[0] for rawp in self.raw_paths])
        self.event = pd.read_csv(self.event_path)
        raws = self.raws.copy()
        for raw in raws:
            ori_sfreq = round(raw.info['sfreq'], 1)
            ori_lfreq = round(raw.info['highpass'], 1)
            ori_hfreq = round(raw.info['lowpass'], 1)
            raw.load_data()
            raw.resample(self.sfreq) if ori_sfreq != raw.info['sfreq'] else None
            raw.filter(self.lfreq, self.hfreq) if self.lfreq != ori_lfreq or self.hfreq != ori_hfreq else None
        self.raws = raws
        del raws
        
    @loading_('align MEG head position')
    def align_head(self, method = 'maxwell', head_idx=0)->None:
        """
        head_idx: int,which run's head position to be the reference
        method: str, maxwell align or directly modify each run's head position
        """
        raise ValueError('one of maxwell, direct') if method not in ['maxwell', 'direct'] else None
        ref_head = self.raws[ref_idx].info['dev_head_t']
        raws = []
        for raw in self.raws:
            raw = self.raws.copy()
            raw = self.maxwell_filter(raw, ref_head) if method == 'maxwell' else raw.info.update({'dev_head_t': ref_head})
            raws.append()
        self.raws = raws
        del raws
    
    @loading_('epoching data')
    def epoching(self)->None:
        sub_epo = []
        for info, raw in zip(self.raws_info, self.raws):
            sub = re.search(r'sub-(\d+)', info).group(1)
            run = re.search(r'run-(\d+)', info).group(1)
            ses = re.search(r'ses-(.*?)_run-', info).group(1)
            event_sub = self.event[(self.event['subject'] == int(sub)) & 
                               (self.event['run'] == int(run)) & 
                               (self.event['session'] == ses)]
            
            events, event_ids = mne.events_from_annotations(raw)
            new_events      = events[events[:,2] == event_ids[event_id]]
            new_event_id    = {self.event_id: event_ids[event_id]}
            
            epochs          = mne.Epochs(
                                raw         = raw,
                                events      = new_events,
                                event_id    = new_event_id,
                                tmin        = self.tmin,
                                tmax        = self.tmax,
                                metadata    = event_sub,
                                # baseline    = (None,0) # use zscore method instead
                                picks       = datatype      
                                )
            sub_epo.append(epochs)
        self.epoched = mne.concatenate_epochs(epochs_list=sub_epo, add_offset=True)
            
            
    def maxwell_filter(raw,ref_head):
        raw_sss = mne.preprocessing.maxwell_filter(raw, 
                                                origin=(0., 0., 0.04),
                                                coord_frame='head',
                                                destination=ref_head,
                                                )
        return raw_sss

    def baseline_correction(self, epochs):
        baselined_epochs = mne.baseline.rescale(data=epochs.get_data(),times=epochs.times,baseline=(None,0),mode='zscore',copy=False)
        epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin,event_id=epochs.event_id, metadata=epochs.metadata)
        return epochs
    
    def run(self,save=False,save_path=None)->None:
        self.load_data()
        self.align_head() if datatype == 'meg' else None
        self.epoching()
        if save:
            self.epoched.save(f'{save_path}/sub-{self.sub}_epo.fif')

#%%
if __name__ == '__main__':
    sub = '01'
    tmin, tmax = -0.1, 0.8
    lfreq, hfreq = 0.1, 40
    sfreq = 200
    datatype = 'eeg'
    event_id = 'stim_on'
    root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/NOD-EEG/derivatives/preprocessed/raw'
    event_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/NOD-EEG/events'
    info = InfoExtraction(root, event_root)
    epochor = Epoching(info.sub_info(sub)['events'], info.sub_info('01')['rawps'], tmin, tmax, sfreq, lfreq, hfreq, datatype)
    epochor.run()

# %%
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import mne
import pandas as pd
from wasabi import msg
mne.use_log_level(verbose='ERROR')
mne.cuda.init_cuda(verbose=True)


def loading_(start='Processing...'):
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
                msg.fail(
                    f'Error in function "{func.__name__}": {str(e)}. Time: {end_time}',
                )
                raise
        return wrapper
    return decorator


class InfoExtraction():
    def __init__(self, root: str, event_root: str) -> None:
        """get all the raw and event under the root directory holding them

        Parameters
        ----------
        root : str
            root directory holding all the raws .fif files
        event_root : str
            root directory holding all the events .csv files

        Returns
        -------
        None
        """
        all_events = {
            f.split('_')[0][-2:]: f'{event_root}/{f}'
            for f in sorted(os.listdir(event_root)) if f.endswith('.csv')
        }

        all_paths = defaultdict(list)
        for p in sorted(os.listdir(root)):
            sub = p.split('_')[0][-2:]
            if p.endswith('.fif'):
                all_paths[sub].append(f"{root}/{p}")
        self.root = root
        self.raw_paths = all_paths
        self.event_paths = all_events
        self.subs = sorted(all_paths.keys())

        events_subs_mapping = all([sub in all_events for sub in all_paths])
        if events_subs_mapping:
            msg.good('all subject consistent in events and raws')
        else:
            msg.fail('subject inconsistent in events and raws')
            print(f'raw path: {self.raw_paths}')
            print(f'event path: {self.event_paths}')
            incorrect_subs = [
                sub for sub in all_paths if sub not in all_events
            ]
            print(f'incorrect subjects: {incorrect_subs}')

    def get_sub_fp(
        self,
        sub: str,
    ) -> dict:
        """get the raw and event paths for a specific subject

        Parameters
        ----------
        sub : str
            subject id

        Returns
        -------
        dict
            raw and event paths for the subject
        """
        return {'rawps': sorted(self.raw_paths[sub]), 'events': self.event_paths[sub]}

    def _repr_html_(self):
        to_show = {
            'root': [self.root],
            'nSubjects': [len(self.subs)],
            'nEvents': [len(self.event_paths)],
        }
        show = pd.DataFrame(to_show).T
        show.columns = ['NOD-InfoExtractor']
        return show.to_html()


class Epoching():
    def __init__(
        self,
        event_csv: str,
        raw_paths: list[str] | str,
        tmin: float,
        tmax: float,
        sfreq: float,
        lfreq: float,
        hfreq: float,
        datatype: str,
        event_id: str,
    ) -> None:
        """make epoched data of one subject, one datatype.

        Parameters
        ----------
        event_csv : str
            path to the event csv file
        raw_paths : list[str] | str
            path to the raw fif files of this subject
        tmin : float
            start time of the epoch
        tmax : float
            end time of the epoch
        sfreq : float
            sampling frequency
        lfreq : float
            low frequency of the filter
        hfreq : float
            high frequency of the filter
        datatype : str
            'meg' or 'eeg'
        event_id : str
            the event id to be extracted
        """
        self.sub = re.search(r'sub-(\d+)_events\.csv', event_csv).group(1)
        msg.divider(f'Epoching-sub{self.sub}_{datatype}')
        self.event_path = event_csv
        self.raw_paths = raw_paths
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.lfreq = lfreq
        self.hfreq = hfreq
        self.datatype = datatype
        self.event_id = event_id

    def run(
        self,
        align_method='info_with_data',
        head_idx=0,
    ) -> mne.Epochs:
        """
        Run full pipeline of epoching.

        Parameters
        ----------
        align_method : str, optional
            Method to align MEG head position, by default 'info_with_data'.
            Options are 'maxwell', 'info_only', 'info_with_data'.
            'maxwell': Use mne's maxwell filter (signal-space separation) to align head position to head_idx raw.
            'info_only': Only update the head position info of the raws based on head_idx raw.
            'info_with_data': Update the head position info of the raws based on head_idx raw and modify the data.

        head_idx : int, optional
            Which run's head position to be the reference, by default 0.

        Returns
        -------
        mne.Epochs
            The concatenated epochs of one subject's all runs.
        """
        self.load_data()
        self.align_head(
            method=align_method,
        ) if self.datatype == 'meg' else None
        self.epoching()

        return self.epoched

    @loading_('loading data & resample/filter if necessary')
    def load_data(self) -> None:
        """load every raw data and resample/filter if necessary"""
        self.raws = [
            mne.io.read_raw_fif(rawp, preload=True)
            for rawp in self.raw_paths
        ]
        self.raws_info = sorted(
            [rawp.split('/')[-1].split('.')[0] for rawp in self.raw_paths],
        )
        self.event = pd.read_csv(self.event_path)
        raws = self.raws.copy()
        for raw in raws:
            ori_sfreq = round(raw.info['sfreq'], 1)
            ori_lfreq = round(raw.info['highpass'], 1)
            ori_hfreq = round(raw.info['lowpass'], 1)
            raw.load_data()
            raw.resample(
                self.sfreq,
            ) if ori_sfreq != raw.info['sfreq'] else None
            raw.filter(
                self.lfreq, self.hfreq,
            ) if self.lfreq != ori_lfreq or self.hfreq != ori_hfreq else None
        self.raws = raws
        del raws

    @loading_('align MEG head position')
    def align_head(
        self,
        method='maxwell',
        head_idx=0,
    ) -> None:
        """Align MEG head position to a reference run."""
        if self.datatype != 'meg':
            raise ValueError('only for MEG data')

        if method not in ['maxwell', 'info_only', 'info_with_data']:
            raise ValueError('one of maxwell, info_only, info_with_data')
        # maxwell not useful for we've already remove ref-channel and
        # disc channel in preprocessing which is
        # ensential for maxwell filter

        ref_head = self.raws[head_idx].info['dev_head_t']
        raws = []
        for raw in self.raws:
            if method == 'maxwell':
                raw = self._maxwell_filter(raw, ref_head)
            elif method == 'info_only':
                raw.info.update({'dev_head_t': ref_head})
            elif method == 'info_with_data':
                raw = self._aligh_head_modify(raw, self.raws[head_idx])
            raws.append(raw)
        self.raws = raws
        del raws

    @loading_('epoching data')
    def epoching(self) -> None:
        """epoching data of all runs and concatenate them"""
        sub_epo = []
        for info, raw in zip(self.raws_info, self.raws):
            sub = re.search(r'sub-(\d+)', info).group(1)
            run = re.search(r'run-(\d+)', info).group(1)
            ses = re.search(r'ses-(.*?)_task-', info).group(1)
            event_sub = self.event[(self.event['subject'] == int(sub)) &
                                   (self.event['run'] == int(run)) &
                                   (self.event['session'] == ses)]

            events, event_ids = mne.events_from_annotations(raw)
            new_events = events[events[:, 2] == event_ids[self.event_id]]
            new_event_id = {self.event_id: event_ids[self.event_id]}

            epochs = mne.Epochs(
                raw=raw,
                events=new_events,
                event_id=new_event_id,
                tmin=self.tmin,
                tmax=self.tmax,
                metadata=event_sub,
                # baseline    = (None,0) # use zscore method instead
                picks=self.datatype,
            )
            sub_epo.append(epochs)
        self.epoched = mne.concatenate_epochs(
            epochs_list=sub_epo, add_offset=True,
        )

    def _maxwell_filter(self, raw, ref_head):
        raw_sss = mne.preprocessing.maxwell_filter(
            raw,
            origin=(0., 0., 0.04),
            coord_frame='head',
            destination=ref_head,
        )
        return raw_sss

    def _aligh_head_modify(self, raw, ref_raw):
        info_from = ref_raw.info
        info_to = raw.info
        map = mne.forward._map_meg_or_eeg_channels(
            info_from, info_to, mode='fast', origin=(0., 0., 0.04),
        )
        data = raw.get_data(picks='meg')
        data_new = map @ data

        raw._data = data_new
        raw.info.update({'dev_head_t': ref_raw.info['dev_head_t']})
        return raw

    def baseline_correction(self, epochs):
        baselined_epochs = mne.baseline.rescale(
            data=epochs.get_data(
            ), times=epochs.times, baseline=(None, 0), mode='zscore', copy=False,
        )
        epochs = mne.EpochsArray(
            baselined_epochs, epochs.info, epochs.events,
            epochs.tmin, event_id=epochs.event_id, metadata=epochs.metadata,
        )
        return epochs


# %%
if __name__ == '__main__':
    sub = '01'
    tmin, tmax = -0.1, 0.8
    lfreq, hfreq = 0.1, 40
    sfreq = 200
    datatype = 'eeg'
    event_id = 'stim_on'
    root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_upload/NOD-EEG/derivatives/cleaned_raw'
    event_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_upload/NOD-EEG/events'
    info = InfoExtraction(root, event_root)
    epochor = Epoching(
        info.get_sub_fp(sub)['events'],
        info.get_sub_fp('01')['rawps'],
        tmin, tmax, sfreq,
        lfreq, hfreq,
        datatype, event_id,
    )
    epochor.run()
    # %%
    datatype = 'meg'
    root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_upload/NOD-MEG/derivatives/cleaned_raw'
    event_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_upload/NOD-MEG/events'
    info = InfoExtraction(root, event_root)
    epochor = Epoching(
        info.get_sub_fp(sub)['events'],
        info.get_sub_fp('01')['rawps'],
        tmin, tmax, sfreq,
        lfreq, hfreq,
        datatype, event_id,
    )
    epochor.run(align_method='info_only')

# %%

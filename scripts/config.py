#%%
from dataclasses import dataclass

@dataclass
class NODConfig:
    root: str
    MEG_bids_root: str
    EEG_bids_root: str
    MEG_events_root: str
    EEG_events_root: str
#%%

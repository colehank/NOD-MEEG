# %%
from __future__ import annotations

import os
from dataclasses import dataclass

from mne_bids import BIDSPath


class NODConfig:
    def __init__(self, root):
        if os.is_dir(root):
            self.root = root
        else:
            raise ValueError('root must be a valid directory')

    def _get_info(self):
        subs = ...
        ses = ...
        tasks = ...
        runs = ...


# %%

# %%
from __future__ import annotations

import mne
import numpy as np
from joblib import delayed
from joblib import Parallel
from scipy.spatial.distance import cdist
from tqdm.autonotebook import tqdm
from tqdm_joblib import tqdm_joblib
from wasabi import msg

from ..utils import get_soi_picks
# %%


class Epo2Rdm:
    def __init__(self, epochs: mne.Epochs) -> None:
        """Initialize the Epo2Rdm class with MNE Epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            The MNE Epochs object containing the data.
        """
        self.epochs = epochs

    def epo2evos(
        self,
        conditions: list[str],
        key: str,
        n_jobs: int,
    ) -> list[mne.Evoked]:
        """Average epochs by specified conditions.

        Parameters
        ----------
        conditions : list[str]
            list of condition names.
        key : str
            Metadata key to select conditions.

        Returns
        -------
        list[mne.Evoked]
            list of averaged Evoked objects ordered by conditions.
        """
        def avg_condition(condition):
            condition_epochs = self.epochs[self.epochs.metadata[key] == condition]
            if not condition_epochs:
                raise ValueError(f"No epochs found for condition: {condition}")
            evoked = condition_epochs.average()
            evoked.comment = condition
            return evoked

        with tqdm_joblib(desc='Processing Conditions', total=len(conditions)):
            evoked_list = Parallel(n_jobs=n_jobs)(
                delayed(avg_condition)(condition) for condition in conditions
            )

        return evoked_list

    # @staticmethod
    # def compute_rdm(
    #     evoked_list: list[mne.Evoked],
    #     fill_diagonal: float = np.nan,
    #     soi: str | None = None,
    #     is_spatiotemporal: bool = False
    # ) -> np.ndarray | list[np.ndarray]:
    #     """Compute Representational Dissimilarity Matrices (RDMs) for each condition.

    #     Parameters
    #     ----------
    #     evoked_list : list[mne.Evoked]
    #         list of Evoked objects representing different conditions.
    #     fill_diagonal : float, optional
    #         Value to fill the diagonal of the RDM. Default is np.nan.
    #     soi : Optional[str], optional
    #         Sensor of Interest (SOI) to select specific channels. Default is None (all sensors).
    #     is_spatiotemporal : bool, optional
    #         If True, compute a spatiotemporal RDM. Otherwise, compute RDMs for each time point. Default is False.

    #     Returns
    #     -------
    #     np.ndarray | list[np.ndarray]
    #         Spatiotemporal RDM if `is_spatiotemporal` is True, else a list of RDMs for each time point.
    #     """
    #     # Select sensors of interest
    #     picks = 'all' if soi is None else soi
    #     soi_picks = get_soi_picks(evoked_list[0], picks)

    #     # Z-score the data across time points
    #     zscored_evos = [Epo2Rdm._zscore_evo(evo, soi_picks) for evo in evoked_list]

    #     if is_spatiotemporal:
    #         # Compute spatiotemporal RDM
    #         sfreq = evoked_list[0].info['sfreq']
    #         onset_sample = int(evoked_list[0].time_as_index(0))  # Convert onset time to sample index
    #         data_matrix = np.array([
    #             evo.data[soi_picks, onset_sample:].flatten()
    #             for evo in zscored_evos
    #         ])
    #         zscored_data = Epo2Rdm._zscore_data(data_matrix)
    #         rdm = np.corrcoef(zscored_data)
    #         np.fill_diagonal(rdm, fill_diagonal)
    #         return rdm
    #     else:
    #         # Compute RDM for each time point
    #         rdms = []
    #         times = evoked_list[0].times
    #         for t_idx in range(len(times)):
    #             data_matrix = np.array([
    #                 evo.data[soi_picks, t_idx]
    #                 for evo in zscored_evos
    #             ])
    #             zscored_data = Epo2Rdm._zscore_data(data_matrix)
    #             rdm = np.corrcoef(zscored_data)
    #             np.fill_diagonal(rdm, fill_diagonal)
    #             rdm = -rdm
    #             rdms.append(rdm)
    #         return rdms

    @staticmethod
    def compute_rdm(
        evoked_list: list[mne.Evoked],
        fill_diagonal: float = np.nan,
        soi: str | None = None,
        is_spatiotemporal: bool = False,
        metric: str = 'correlation',
    ) -> np.ndarray | list[np.ndarray]:
        """Compute Representational Dissimilarity Matrices (RDMs) for each condition.

        Parameters
        ----------
        evoked_list : list[mne.Evoked]
            List of Evoked objects representing different conditions.
        fill_diagonal : float, optional
            Value to fill the diagonal of the RDM. Default is np.nan.
        soi : Optional[str], optional
            Sensor of Interest (SOI) to select specific channels. Default is None (all sensors).
        is_spatiotemporal : bool, optional
            If True, compute a spatiotemporal RDM. Otherwise, compute RDMs for each time point. Default is False.
        metric : str, optional
            The distance metric to use for computing the RDM. Options include 'euclidean', 'cosine', 'correlation', etc.
            Default is 'correlation'.

        Returns
        -------
        np.ndarray or list[np.ndarray]
            Spatiotemporal RDM if `is_spatiotemporal` is True, else a list of RDMs for each time point.
        """
        # Select sensors of interest
        picks = 'all' if soi is None else soi
        soi_picks = get_soi_picks(evoked_list[0], picks)

        # Z-score the data across time points
        zscored_evos = [
            Epo2Rdm._zscore_evo(
                evo, soi_picks,
            ) for evo in evoked_list
        ]

        if is_spatiotemporal:
            # Compute spatiotemporal RDM
            sfreq = evoked_list[0].info['sfreq']
            # Convert onset time to sample index
            onset_sample = int(evoked_list[0].time_as_index(0))
            data_matrix = np.array([
                evo.data[soi_picks, onset_sample:].flatten()
                for evo in zscored_evos
            ])
            zscored_data = Epo2Rdm._zscore_data(data_matrix)
            # Compute RDM using specified metric
            if metric == 'pearson':
                rdm = - np.corrcoef(zscored_data)
            else:
                rdm = cdist(
                    zscored_data,
                    zscored_data,
                    metric=metric,
                )
            np.fill_diagonal(rdm, fill_diagonal)
            return rdm
        else:
            # Compute RDM for each time point
            rdms = []
            times = evoked_list[0].times
            for t_idx in range(len(times)):
                data_matrix = np.array([
                    evo.data[soi_picks, t_idx]
                    for evo in zscored_evos
                ])
                zscored_data = Epo2Rdm._zscore_data(data_matrix)
                # Compute RDM using specified metric
                if metric == 'pearson':
                    rdm = - np.corrcoef(zscored_data)
                else:
                    rdm = cdist(
                        zscored_data,
                        zscored_data,
                        metric=metric,
                    )
                np.fill_diagonal(rdm, fill_diagonal)
                rdms.append(rdm)
            return rdms

    @staticmethod
    def _zscore_data(data: np.ndarray) -> np.ndarray:
        """Z-score the data across the first axis.

        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_conditions, n_features).

        Returns
        -------
        np.ndarray
            Z-scored data.
        """
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, ddof=1, keepdims=True)
        zscored = (data - mean) / std
        return zscored

    @staticmethod
    def _zscore_evo(
        evoked: mne.Evoked,
        picks: list[int],
    ) -> mne.Evoked:
        """Z-score each time point of an Evoked object for specified sensors.

        Parameters
        ----------
        evoked : mne.Evoked
            The Evoked object to be z-scored.
        picks : list[int]
            list of sensor indices to apply z-scoring.

        Returns
        -------
        mne.Evoked
            Z-scored Evoked object.
        """
        data = evoked.data[picks, :]
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, ddof=1, keepdims=True)
        zscored_data = (data - mean) / std
        zscored_evo = evoked.copy()
        zscored_evo.data[picks, :] = zscored_data
        return zscored_evo


def make_grand_evos(
    epochs: mne.Epochs,
    conditions: list[str],
    key: str,
    n_jobs: int,
) -> mne.Evoked:
    Epochs2RDM = Epo2Rdm(epochs)
    evoked_list = Epochs2RDM.epo2evos(conditions, key, n_jobs)

    return evoked_list


def evos2rdm(
    evos: list[mne.Evoked],
    soi: list[str],
    is_spatiotemporal: bool,
    metric: str = 'correlation',
    fill_diagonal: float = np.nan,
) -> np.array:

    rdm = Epo2Rdm.compute_rdm(
        evos, fill_diagonal,
        soi, is_spatiotemporal, metric,
    )
    return rdm


def generate_rdms(
    evos: list[mne.Evoked],
    sois: list[str],
    metric: str,
    is_spatiotemporal: bool,
    fill_diagonal: float = np.nan,
    n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    def process_soi(soi):
        return soi, evos2rdm(
            evos=evos,
            soi=soi,
            is_spatiotemporal=is_spatiotemporal,
            fill_diagonal=fill_diagonal,
            metric=metric,
        )

    with tqdm_joblib(desc='Processing SOIs', total=len(sois)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_soi)(soi) for soi in sois
        )

    soi_rdms = {soi: rdm for soi, rdm in results}
    return soi_rdms


def make_grand_epo(
    epo_ps: list[str],
    align_by: str,
    align_to: int,
    n_jobs: int,
) -> mne.Epochs:
    def read_epoch(epo_p):
        return mne.read_epochs(epo_p)

    def _align_epoch(epo, info_from):
        info_to = epo.info
        map = mne.forward._map_meg_or_eeg_channels(
            info_from, info_to, mode='fast', origin=(0., 0., 0.04),
        )
        data = epo.get_data()
        data_new = map @ data

        epo._data = data_new
        epo.info.update({'dev_head_t': info_from['dev_head_t']})
        return epo

    with tqdm_joblib(desc='Reading epochs', total=len(epo_ps)):
        epos = Parallel(n_jobs=n_jobs)(
            delayed(read_epoch)(epo_p) for epo_p in epo_ps
        )

    ref_head = epos[align_to]
    if align_by == 'info':
        for epo in tqdm(epos, desc='Aligning epochs'):
            epo.info['dev_head_t'] = ref_head.info['dev_head_t']
        with msg.loading('  Concatenating epochs...'):
            grand_epo = mne.concatenate_epochs(epos)
        msg.good('epochs concatenated')
        return grand_epo

    elif align_by == 'info_with_data':
        info_from = ref_head.info
        with tqdm_joblib(desc='Aligning epochs', total=len(epos)):
            epos_aligned = Parallel(n_jobs=n_jobs)(
                delayed(_align_epoch)(epo, ref_head.info) for epo in epos
            )

        with msg.loading('  Concatenating epochs...'):
            grand_epo = mne.concatenate_epochs(epos_aligned)
        msg.good('epochs concatenated')
        return grand_epo

    else:
        raise ValueError("align_by must be either 'info' or 'info_with_data'")
# %%

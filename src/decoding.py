#%%
import numpy as np
from dataclasses import dataclass
import mne
from mne.io import BaseRaw
from mne import BaseEpochs
from mne.decoding import UnsupervisedSpatialFilter, Scaler, SlidingEstimator, Vectorizer, cross_val_multiscore

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

if __name__ == "__main__":
    from utils import get_soi_picks
else:
    from .utils import get_soi_picks

#%%
class DecodingEpochs:
    def __init__(
        self,
        epochs: list[BaseEpochs, BaseEpochs],
        pick_type:str,
        random_state: int = 97,
        n_jobs: int = 8
        ):
        self.pick_type = pick_type
        epos = self._preprpcess_epochs(epochs)
        
        self.info = epos[0].info
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        X = np.concatenate(
            (epos[0]._data, epos[1]._data),
            axis=0
            )
        y = np.concatenate(
            (
                np.ones(len(epos[0])),
                np.zeros(len(epos[1])),
            )
            )
        
        
        self.X, self.y = shuffle(
            X, y, random_state=self.random_state
            )
    
    
    def fit(
        self,
        metric: str,
        cv: int = 10,
        pca: bool = True,
        C: float = 0.5,
        tol: float = 1e-3,
        pca_threshold: float = 0.95,
        ) -> np.ndarray:
        
        Scaler = mne.decoding.Scaler(info=self.info)
        X = Scaler.fit_transform(self.X)
        
        X = self._pca_on_channel(
            X, pca_threshold) if pca else X
        y = self.y
        
        if cv == 'loo':
            cv = LeaveOneOut()
        else:
            cv = StratifiedKFold(n_splits=cv, 
                                random_state=self.random_state,
                                shuffle=True)
        clf = make_pipeline(
            Vectorizer(),
            LinearSVC(C=C, 
                random_state=self.random_state,
                class_weight='balanced',
                tol=tol,
                max_iter=99999,
                )
            )
        
        time_decod = SlidingEstimator(
            clf, 
            scoring = metric, 
            n_jobs = self.n_jobs,
            verbose = None,
            )
        
        scores = cross_val_multiscore(
            time_decod, 
            X, y, 
            cv=cv,
            n_jobs=1,
            verbose=None,
            )
        
        
        return scores.mean(axis = 0)
        
    def _preprpcess_epochs(
        self, 
        epos: list[BaseEpochs],
        drop_chans: list[str] = ['M1', 'M2']
        ) -> list[BaseEpochs]:
        epochs = []
        for epo in epos:
            if any(chan in epo.ch_names for chan in drop_chans):
                epo.drop_channels(ch_names=drop_chans)
            epo = epo.pick(
                get_soi_picks(epo, self.pick_type)
                )
            epochs.append(epo)
        return epochs

    @staticmethod
    def _pca_on_channel(
        X: np.ndarray,
        threshold: float
        ) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                'X should be 3D array, (n_epochs, n_channels, n_times)'
                )
            
        PCA_ = UnsupervisedSpatialFilter(PCA(threshold), average=False)
        return PCA_.fit_transform(X)
    
        
    def _repr_html_(self):
        import pandas as pd
        to_show = {
            'n_samplele': len(self.X),
            'sensorType': self.pick_type,
            'nChannels': len(self.info['ch_names']),
            'nClasses': len(np.unique(self.y)),
            'nJobs': self.n_jobs,
            'randomState': self.random_state,
            }
        to_show = pd.DataFrame(to_show, index=[0]).T
        to_show.columns = ['SVMClassifier']
        return to_show.to_html()

def cls_2epo(
    epos: list[BaseEpochs, BaseEpochs],
    n_sample: int = 1000,
    cv: int | str = 3,
    SOI:str = 'OT',
    metric:str = 'accuracy',
    n_jobs:int = 1,
    C:float = 0.5,
    tol:float = 1e-1,
    pca_threshold:float | int = 0.95,
    pca:bool = False,
    plot:bool = False,
    ) -> np.ndarray:
    """using SVM to classify two epochs.
    
    Parameters
    ----------
    epos : list[BaseEpochs, BaseEpochs]
        two epochs to be classified.
    n_sample : int, optional
        number of samples to be used, by default 1000
    cv : int | str, optional
        cross validation method, by default 3
    SOI : str, optional
        sensor of interest, by default 'OT', occipital sensors& temporal sensors
    metric : str, optional
        the scoring metric to be used, by default 'accuracy'
    n_jobs : int, optional
        number of jobs to run in parallel, by default 64
    C : float, optional
        the penalty parameter of the error term, by default 0.5
    tol : float, optional
        tolerance for stopping criteria, by default 1e-1
    pca_threshold : float | int, optional
        if pca is True, the threshold of pca, by default 0.95
    pca : bool, optional
        if True, use pca to reduce dimension, by default False
    
    Returns
    -------
    np.ndarray
        the score of the classification
    """
    assert len(epos) == 2, 'epos should be a list of two epochs'
    assert n_sample in ['all', 'min'] or isinstance(n_sample, int) and n_sample > 0, \
        'n_sample should be a positive int or "all" or "min"'
    assert cv in ['loo', 'all'] or isinstance(cv, int), \
        'cv should be an int or "loo" or "all"'
    
    if n_sample == 'all':
        classfier = DecodingEpochs(epos, SOI, n_jobs=n_jobs)
    elif n_sample == 'min':
        min_samples = min(len(epos[0]), len(epos[1]))
        classfier = DecodingEpochs(
            [epos[0][:min_samples], epos[1][:min_samples]], 
            SOI, 
            n_jobs=n_jobs,
        )
    else:
        classfier = DecodingEpochs(
            [epos[0][:n_sample], epos[1][:n_sample]], 
            SOI, 
            n_jobs=n_jobs,
        )
    
    score = classfier.fit(
        metric,
        cv=cv,
        pca=pca,
        C=C,
        tol=tol,
        )
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(epos[0].times, score, label="score")
        ax.axhline(0.5, color="k", linestyle="--", label="chance")
        ax.set_xlabel("Times")
        ax.set_ylabel("Performance")
        ax.legend()
        ax.axvline(0.0, color="k", linestyle="-")
        ax.set_title("Sensor space decoding")
        
    return score
    
#%%
if __name__ == '__main__':
    def cls_2epo(
        epos: list[BaseEpochs, BaseEpochs],
        n_sample: int = 1000,
        cv: int | str = 3,
        SOI:str = 'OT',
        metric:str = 'accuracy',
        n_jobs:int = 64,
        C:float = 0.5,
        tol:float = 1e-1,
        pca_threshold:float | int = 0.95,
        pca:bool = False,
        plot:bool = False,
        ) -> np.ndarray:
        """using SVM to classify two epochs.
        
        Parameters
        ----------
        epos : list[BaseEpochs, BaseEpochs]
            two epochs to be classified.
        n_sample : int, optional
            number of samples to be used, by default 1000
        cv : int | str, optional
            cross validation method, by default 3
        SOI : str, optional
            sensor of interest, by default 'OT', occipital sensors& temporal sensors
        metric : str, optional
            the scoring metric to be used, by default 'accuracy'
        n_jobs : int, optional
            number of jobs to run in parallel, by default 64
        pca_threshold : float | int, optional
            if pca is True, the threshold of pca, by default 0.95
        pca : bool, optional
            if True, use pca to reduce dimension, by default False
        
        Returns
        -------
        np.ndarray
            the score of the classification
        """
        if n_sample == 'all':
            classfier = DecodingEpochs(epos, SOI, n_jobs=n_jobs)
        else:
            classfier = DecodingEpochs(
                [epos[0][:n_sample], epos[1][:n_sample]], 
                SOI, 
                n_jobs=n_jobs,
            )
        
        score = classfier.fit(
            metric,
            cv=cv,
            pca=pca,
            C=C,
            tol=tol,
            )
        
        fig, ax = plt.subplots()
        ax.plot(epos[0].times, score, label="score")
        ax.axhline(0.5, color="k", linestyle="--", label="chance")
        ax.set_xlabel("Times")
        ax.set_ylabel("AUC")
        ax.legend()
        ax.axvline(0.0, color="k", linestyle="-")
        ax.set_title("Sensor space decoding")
        return score


    SUB = '01'
    Eepo = mne.read_epochs(
        f'/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_upload/NOD-EEG/derivatives/preprocessed/epochs/sub-{SUB}_eeg_epo.fif'
        )
    Mepo = mne.read_epochs(
        f'/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_upload/NOD-MEG/derivatives/preprocessed/epochs/sub-{SUB}_meg_epo.fif'
        )
    
    ani_epo = Mepo[Mepo.metadata['stim_is_animate'] == True]
    inani_epo = Mepo[Mepo.metadata['stim_is_animate'] == False]
    epos = [ani_epo, inani_epo]
    score = cls_2epo(
        epos, 
        n_sample=1000, 
        cv=10, 
        SOI = 'OT', 
        metric='accuracy',
        n_jobs=64,
        C=0.5,
        tol=0.01,
        pca=False,
        plot=True,
        )
#%%

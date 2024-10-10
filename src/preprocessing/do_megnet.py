#%%
import io
import contextlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse
import numpy as np
import pandas as pd
import scipy.io
idx = pd.IndexSlice
from scipy.io import loadmat
import json
import pickle
import scipy.stats as stats
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from joblib import Parallel, delayed

from tqdm import TqdmExperimentalWarning
import warnings
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

from tqdm.autonotebook import tqdm
from tqdm_joblib import tqdm_joblib
#%%
def fGetStartTimesOverlap(intInputLen, intModelLen=15000, intOverlap=3750):
    """
    model len is 60 seconds at 250Hz = 15000
    overlap len is 15 seconds at 250Hz = 3750
    """
    lStartTimes = []
    intStartTime = 0
    while intStartTime+intModelLen<=intInputLen:
        lStartTimes.append(intStartTime)
        intStartTime = intStartTime+intModelLen-intOverlap
    return lStartTimes

def fPredictChunkAndVoting(kModel, lTimeSeries, arrSpatialMap, arrY, intModelLen=15000, intOverlap=3750):
    """
    This function is designed to take in ICA time series and a spatial map pair and produce a prediction useing a trained model.
    The time series will be split into multiple chunks and the final prediction will be a weighted vote of each time chunk.
    The weight for the voting will be determined by the manout of time and overlap each chunk has with one another.
    For example if the total lenght of the scan is 50 seconds, and the chunks are 15 seconds long with a 5 second overlap:
        The first chunk will be the only chunk to use the first 10 seconds, and one of two chunks to use the next 5 seconds.
            Thus   

    :param kModel: The model that will be used for the predictions on each chunk. It should have two inputs the spatial map and time series respectivley
    :type kModel: a keras model
    :param lTimeSeries: The time series for each scan (can also be an array if all scans are the same lenght)
    :type lTimeSeries: list or array (if each scan is a different length, then it needs to be a list)
    :param arrSpatialMap: The spatial maps (one per scan)
    :type arrSpatialMap: numpy array
    :param intModelLen: The lenght of the time series in the model, defaults to 15000
    :type intModelLen: int, optional
    :param intOverlap: The lenght of the overlap between scans, defaults to 3750
    :type intOverlap: int, optional
    """
    #empty list to hold the prediction for each component pair
    lPredictionsVote = []
    lGTVote = []

    lPredictionsChunk = []
    lGTChunk = []

    i = 0
    for arrScanTimeSeries, arrScanSpatialMap, arrScanY in zip(lTimeSeries, arrSpatialMap, arrY):
        intTimeSeriesLen = arrScanTimeSeries.shape[0]
        lStartTimes = fGetStartTimesOverlap(intTimeSeriesLen, intModelLen=intModelLen, intOverlap=intOverlap)

        if lStartTimes[-1]+intModelLen <= intTimeSeriesLen:
            lStartTimes.append(arrScanTimeSeries.shape[0]-intModelLen)


        lTimeChunks = [[x,x+intModelLen] for x in lStartTimes]
        dctTimeChunkVotes = dict([[x,0] for x in lStartTimes])
        for intT in range(intTimeSeriesLen):
            lChunkMatches = [x <= intT < x+intModelLen for x in dctTimeChunkVotes.keys()]
            intInChunks = np.sum(lChunkMatches)
            for intStartTime, bTruth in zip(dctTimeChunkVotes.keys(), lChunkMatches):
                if bTruth:
                    dctTimeChunkVotes[intStartTime]+=1.0/intInChunks

        #predict
        dctWeightedPredictions = {}
        for intStartTime in dctTimeChunkVotes.keys():
            lPrediction = kModel.predict([np.expand_dims(arrScanSpatialMap,0),
                                        np.expand_dims(np.expand_dims(arrScanTimeSeries[intStartTime:intStartTime+intModelLen],0),-1)],
                                         verbose=0)
            lPredictionsChunk.append(lPrediction)
            lGTChunk.append(arrScanY)
            
            dctWeightedPredictions[intStartTime] = lPrediction*dctTimeChunkVotes[intStartTime]

        # arrScanPrediction = np.stack(dctWeightedPredictions.values())
        arrScanPrediction = np.stack(list(dctWeightedPredictions.values()))
        arrScanPrediction = arrScanPrediction.mean(axis=0)
        arrScanPrediction = arrScanPrediction/arrScanPrediction.sum()
        lPredictionsVote.append(arrScanPrediction)
        lGTVote.append(arrScanY)
        
        print(f"\r{i+1}/{arrY.shape[0]}",end = '', flush=True)
        i+=1
    return np.stack(lPredictionsVote), np.stack(lGTVote), np.stack(lPredictionsChunk), np.stack(lGTChunk)




def fPredictChunkAndVoting(kModel, lTimeSeries, arrSpatialMap, arrY, intModelLen=15000, intOverlap=3750, n_jobs=-1):
    """
    并行化版本：使用 joblib 并行处理每个组件，并结合 tqdm 显示进度条。
    """
    def process_scan(arrScanTimeSeries, arrScanSpatialMap, arrScanY):
        intTimeSeriesLen = arrScanTimeSeries.shape[0]
        lStartTimes = fGetStartTimesOverlap(intTimeSeriesLen, intModelLen=intModelLen, intOverlap=intOverlap)

        if lStartTimes[-1] + intModelLen <= intTimeSeriesLen:
            lStartTimes.append(arrScanTimeSeries.shape[0] - intModelLen)

        lTimeChunks = [[x, x + intModelLen] for x in lStartTimes]
        dctTimeChunkVotes = dict([[x, 0] for x in lStartTimes])
        for intT in range(intTimeSeriesLen):
            lChunkMatches = [x <= intT < x + intModelLen for x in dctTimeChunkVotes.keys()]
            intInChunks = np.sum(lChunkMatches)
            for intStartTime, bTruth in zip(dctTimeChunkVotes.keys(), lChunkMatches):
                if bTruth:
                    dctTimeChunkVotes[intStartTime] += 1.0 / intInChunks

        
        dctWeightedPredictions = {}
        lPredictionsChunk = []
        lGTChunk = []

        for intStartTime in dctTimeChunkVotes.keys():
            lPrediction = kModel.predict([np.expand_dims(arrScanSpatialMap, 0),
                                          np.expand_dims(np.expand_dims(arrScanTimeSeries[intStartTime:intStartTime + intModelLen], 0), -1)],
                                         verbose=0)
            lPredictionsChunk.append(lPrediction)
            lGTChunk.append(arrScanY)
            dctWeightedPredictions[intStartTime] = lPrediction * dctTimeChunkVotes[intStartTime]

        arrScanPrediction = np.stack(list(dctWeightedPredictions.values())).mean(axis=0)
        arrScanPrediction = arrScanPrediction / arrScanPrediction.sum()

        return arrScanPrediction, arrScanY, lPredictionsChunk, lGTChunk

    with tqdm_joblib(desc="ICA Predicting", total=len(lTimeSeries), leave=False) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(delayed(process_scan)(arrScanTimeSeries, arrScanSpatialMap, arrScanY)
                                          for arrScanTimeSeries, arrScanSpatialMap, arrScanY in zip(lTimeSeries, arrSpatialMap, arrY))
    
    lPredictionsVote, lGTVote, lPredictionsChunk, lGTChunk = zip(*results)

    return (np.stack(lPredictionsVote), 
            np.stack(lGTVote), 
            np.vstack([np.stack(chunks) for chunks in lPredictionsChunk]), 
            np.vstack([np.stack(gt) for gt in lGTChunk]))

def fPredictICA(strSubjectICAPath, strModelPath, Ncomp = 100,strOutputDir=None, strOutputType='list', n_jobs=8):
    """ 
        Predict Independent Component Analysis (ICA) labels using a pre-trained model.
        This function loads ICA time series and spatial maps, ensures their compatibility,
        and uses a pre-trained model to predict labels for each ICA component. The predictions
        can be saved to a specified directory.
        Parameters:
        -----------
        strSubjectICAPath : str
            Path to the directory containing ICA time series and spatial maps.
        strModelPath : str
            Path to the pre-trained model file.
        Ncomp : int, optional
            Number of ICA components to process (default is 100).
        strOutputDir : str, optional
            Directory to save the predicted labels (default is None, which means no saving).
        strOutputType : str, optional
            Format of the output predictions. Can be 'list' or 'array' (default is 'list').
        Returns:
        --------
        to_return : numpy.ndarray
            Predicted labels for each ICA component. The format depends on `strOutputType`:
            - If 'list', returns a 1D array of predicted labels.
            - If 'array', returns a 2D array of prediction probabilities.
        Raises:
        -------
        ValueError
            If the loaded data does not have the correct dimensions.
        Notes:
        ------
        - The function expects the ICA time series to be stored in 'ICATimeSeries.mat' and
          the spatial maps to be stored in 'component{i}.mat' files within `strSubjectICAPath`.
        - The spatial maps are expected to have a shape of [N, 120, 120, 3].
        - The time series should have at least 15000 samples.
    """
    #loading the data is from our Brainstorm Pipeline, it may require some minor edits based on how the data is saved.
    #load the time seris and the spatial map
    arrTimeSeries = scipy.io.loadmat(os.path.join(strSubjectICAPath,'ICATimeSeries.mat'))['arrICATimeSeries'].T
    arrSpatialMap = np.array([scipy.io.loadmat(os.path.join(strSubjectICAPath,f'component{i}.mat'))['array'] for i in range(1,Ncomp+1)])
    #crop the spatial map to remove additional pixels
    # arrSpatialMap = arrSpatialMap[:,30:-30,15:-15,:]
    #ensure the data is compatable
    try:
        assert arrTimeSeries.shape[0] == arrSpatialMap.shape[0], "The number of time series should be the same as the number of spatial maps"
        assert arrSpatialMap.shape[1:] == (120, 120, 3), "The spatial maps should have a shape of [N, 120, 120, 3]"
        assert arrTimeSeries.shape[1] >= 15000, "The time series need to be at least 60 seconds with a sample rate of 250Hz (60*250=15000)"
    except AssertionError as e:
        raise ValueError(f"The data does not have the correct dimensions: {str(e)}")
        
    # custom_objects = {
    # 'Addons>F1Score': tfa.metrics.F1Score
    #                     }
    KModel = load_model(strModelPath)
    # KModel = keras.layers.TFSMLayer('/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_codes/utils/megnet_model/megnet-enigma_model', call_endpoint='serving_default')
    
    #use the vote chunk prediction function to make a prediction on each input
    output = fPredictChunkAndVoting(KModel, 
                                    arrTimeSeries, 
                                    arrSpatialMap, 
                                    np.zeros((Ncomp,3)), #the code expects the Y values as it was used for performance, just put in zeros as a place holder.
                                    intModelLen=15000, 
                                    intOverlap=3750,
                                    n_jobs=n_jobs)
    arrPredicionsVote, arrGTVote, arrPredictionsChunk, arrGTChunk = output

    #format the predictions
    if strOutputType.lower() == 'array':
        to_return = arrPredicionsVote[:,0,:]
    else:
        to_return = arrPredicionsVote[:,0,:].argmax(axis=1)
    
    #save the predictions if path is given
    if not strOutputDir is None:
        strOutputPath = os.path.join(strOutputDir,'ICA_component_lables.txt')
        np.savetxt(strOutputPath, to_return)

    return to_return
#%%
if __name__ == '__main__':
    classes = {}
            
    megnet_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_data/NOD-MEG/derivatives/preprocessed/ICA/megnet'
    # model = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_codes/utils/megnet_model/MEGnet_final_model.h5'
    model = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG/models/megnet_enigma.keras'
    # model = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG/models/megnet_'

    runps = sorted([f'{megnet_root}/{i}' for i in os.listdir(megnet_root)][:2])
    for runp in runps:
        if runp.split('/')[-1] in classes.keys():
            continue
        else:
            out = fPredictICA(runp,model)
            classes[runp.split('/')[-1]] = list(out)

    with open(f'{megnet_root}/classes.json', 'w') as f:
        json.dump(classes, f, default=str, indent=4)
#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from EDMD.hua import approximate_koopman

def loadData(filepath): 
    """
    loads our data. assumes they are stored in .csv files,
    using latin1 encoding
    first column of the csv is time, second column is signal
    """
    data = pd.read_csv(filepath,encoding='latin1',usecols=[0,1])
    time = np.array(data.iloc[:,0])
    signal = np.array(data.iloc[:,1])

    dt = time[1] - time[0]
    fs = 1/dt

    return time,signal, fs

def splitData(time,signal,fs,num_segments = 10, train_segment_length = 6,test_segment_length=1):
    """ 
    takes in time, signal, and sampling rate of data as first arguments
    also takes in: number of train/test segments we want (integer)
    and train segment length (in seconds) and test segment length (in seconds)
    test segment will always follow directly after train segment!
    """
    allSegs = []

    nPoints = len(signal)
    maxStartInd = nPoints - (train_segment_length + test_segment_length)*fs

    for segs in num_segments:

        data = {}

        start = np.random.choice(maxStartInd,1)
        endTrain = start + train_segment_length*fs
        endTest = endTrain + test_segment_length*fs

        trainTime = time[start:endTrain]
        trainSignal = signal[start:endTrain]

        testTime = time[endTrain:endTest]
        testSignal = signal[endTrain:endTest]

        data["train"] = (trainTime,trainSignal)
        data["test"] = (testTime,testSignal)

        allSegs.append(data)

    return data

def calcMetrics(xhat,x):
    """
    takes as input predictions (xhat), data (x). computes and returns
    MSE, l2 norm, and linf of error between true, predicted data.
    """
    err = xhat - x
    mse = np.nanmean(err**2)
    l2 = np.sqrt(np.sum(err ** 2))
    linf = np.amax(err)

    return mse,l2,linf




    
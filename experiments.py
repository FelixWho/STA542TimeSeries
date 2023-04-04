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
    maxStartInd = nPoints - int((train_segment_length + test_segment_length)*fs)

    for segs in range(num_segments):

        data = {}

        start = np.random.choice(maxStartInd,1)[0]
        
        endTrain = start + int(train_segment_length*fs)
        endTest = endTrain + int(test_segment_length*fs)
        
        trainSignal = signal[start:endTrain]
        testSignal = signal[endTrain:endTest]

        data['train'] = trainSignal
        data['test'] = testSignal
        allSegs.append(data)

    return allSegs

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

def forecast(x,L,HOP,extK,extM):

    """
    takes as input x: signal to be forecasted
    L: number of samples to be forecasted
    HOP: subsampling rate for forecasting
    extK: size of dataset for forecasting
    extM: length of segments used for forecasting
    """

    sigma2 = 200
    X = np.zeros((int(np.ceil(extM/HOP)),extK)) # sets up matrix for EDMD estimation
    print(X.shape)
    for kk in range(extK):
        start = -1-extK-extM+kk
        end = -1-extK+kk-1
       
        data = x[start:HOP:end]
        #### TO DO ####
        # fix this part.
        ###############

        X[:,kk] = x[-1-extK-extM+kk:HOP:-1-extK+kk-1]
    Y = np.hstack([X[:,2:], x[-1-extM+1:HOP:]])

    [Xi,mu,phix] = approximate_koopman(X,Y,sigma2)

    Z = np.zeros((np.ceil(extM/HOP),L))
    tmp = phix.T
    for kk in range(L):
        tmp = mu * tmp
        Z[:,kk] = tmp
    Z = np.real(Xi.T @ Z)
    xext = Z[-1,:].T

    return xext

def plotMetrics():
    pass


if __name__ == "__main__":

    
    timeEEG,sigEEG,fsEEG = loadData("F4M1.csv") 
    timePleth,sigPleth,fsPleth = loadData("Pleth.csv") 
    
    HOP = 1
    extSEC = 0.1 # extension length in seconds
    assert fsPleth == fsEEG
    L = np.round(extSEC * fsEEG)
    extM = 750
    extK = int(np.round( 3.75*L )) # number of points to estimate A / size of datasets
    extKSecs = ( 3.75*L/fsEEG )
    eegData = splitData(timeEEG,sigEEG,fsEEG,\
                        train_segment_length=extKSecs,\
                        test_segment_length=extSEC)
    plethData = splitData(timePleth,sigPleth,fsPleth,\
                        train_segment_length=extKSecs,\
                        test_segment_length=extSEC)

    eegStats = []
    plethStats = []

    for seg in eegData:

        xExt = forecast(seg['train'],L,HOP,extK,extM)

        metrics = calcMetrics(xExt,seg['test']) 
        eegStats.append(metrics)

    for seg in plethData:

        xExt = forecast(seg['train'],L,HOP,extK,extM)

        metrics = calcMetrics(xExt,seg['test']) 
        plethStats.append(metrics)
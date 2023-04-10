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

def splitData(time,signal,fs,num_segments = 5, train_segment_length = 6,test_segment_length=1):
    """ 
    takes in time, signal, and sampling rate of data as first arguments
    also takes in: number of train/test segments we want (integer)
    and train segment length (in seconds) and test segment length (in seconds)
    test segment will always follow directly after train segment!
    """
    allSegs = []

    nPoints = len(signal)
    maxStartInd = nPoints - int((train_segment_length + test_segment_length)*fs)
    #print(fs)
    #print(nPoints)
    for segs in range(num_segments):

        data = {}

        start = np.random.choice(maxStartInd,1)[0]
        #print(start)
        endTrain = start + int(train_segment_length*fs)
        #print(train_segment_length * fs)
        #print(endTrain)
        endTest = endTrain + int(test_segment_length*fs)
        
        trainSignal = signal[start:endTrain]
        testSignal = signal[endTrain:endTest]

        #print(trainSignal.shape)
        #print(testSignal.shape)
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

def plotMetrics():
    pass

def forecast(x,L,HOP,extK,extM):

    """
    takes as input x: signal to be forecasted
    L: number of samples to be forecasted
    HOP: subsampling rate for forecasting
    extK: size of dataset for forecasting
    extM: length of segments used for forecasting
    """

    sigma2 = 100
    X = np.zeros((int(np.ceil(extM/HOP)),extK)) # sets up matrix for EDMD estimation
    #x = np.hstack([xTrain,xTest])
    

    ## of length at least extM+ 2*extK
    for kk in range(extK):
        start = -extK - extM +kk
        end = kk - extK
        #print(start)
        #print(end)
        #
        #### TO DO ####
        # fix this part.
        ###############
        #print(start)
        #print(end)
        #print(x.shape)
        data = x[start:end:HOP]
        #if len(data) < extM: print(kk)
        #print(HOP)

        X[:,kk] = x[start:end:HOP]


    A = X[:,1:]
    B = x[-extM::HOP,None]
    
    Y = np.hstack([A, B])
    
    [Xi,mu,phix] = approximate_koopman(X,Y,sigma2)

    Z = np.zeros((round(np.ceil(extM/HOP)),L))
    tmp = phix.T
    for kk in range(L):
        tmp = mu * tmp
        Z[:,kk] = tmp
    Z = np.real(Xi.T @ Z)
    xext = Z[-1,:].T

    return xext


if __name__ == "__main__":

    
    timeEEG,sigEEG,fsEEG = loadData("F4M1.csv") 
    timePleth,sigPleth,fsPleth = loadData("Pleth.csv") 
    
    HOP = 1
    extSEC = 5 # extension length in seconds
    assert fsPleth == fsEEG
    L = round(extSEC * fsEEG)
    extM = round(1.5 * L)
    extK = round( 2.5*extM ) # number of points to estimate A / size of datasets
    extKSecs = ( extK/fsEEG )
    extMSecs = (extM/fsEEG)
    eegData = splitData(timeEEG,sigEEG,fsEEG,\
                        train_segment_length=extKSecs + extMSecs,\
                        test_segment_length=extSEC)
    plethData = splitData(timePleth,sigPleth,fsPleth,\
                        train_segment_length=extKSecs + extMSecs,\
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
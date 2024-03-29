import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from EDMD.window import approximate_koopman
from EDMD.robust import RobustEDMD

def loadData(filepath): 
    """
    loads our data. assumes they are stored in .csv files,
    using latin1 encoding
    first column of the csv is time, second column is signal
    """
    data = pd.read_csv(filepath,encoding='latin1',usecols=[0,1])
    time = np.array(data.iloc[:,0])
    signal = np.array(data.iloc[:,1])

    mu = np.nanmean(signal)
    sigma = np.nanstd(signal)
    #z-scoring data
    signal = (signal - mu)/sigma


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

def plotMetrics(listOfMetrics):

    #plt.rcParams["text.usetex"] = True
    mses = [ls[0] for ls in listOfMetrics]
    l2s = [ls[1] for ls in listOfMetrics]
    linfs = [ls[2] for ls in listOfMetrics]
    

    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(10,10))

    meanMSE = np.nanmean(mses)
    sdMSE = np.nanmean(mses)
    meanl2s = np.nanmean(l2s)
    sdl2s = np.nanmean(l2s)
    meanInfs = np.nanmean(linfs)
    sdInfs = np.nanmean(linfs)

    axs.errorbar([0,1,2],[meanMSE,meanl2s,meanInfs],\
                    yerr=[sdMSE,sdl2s,sdInfs])
    axs.set_xticklabels(["MSE","L2", "LInf"])
    axs.set_ylabel("Error")

    plt.savefig("test.png")


def forecast(x,L,HOP,extK,extM,backward=False):

    """
    takes as input x: signal to be forecasted
    L: number of samples to be forecasted
    HOP: subsampling rate for forecasting
    extK: size of dataset for forecasting
    extM: length of segments used for forecasting

    can also do this backwards? it's in HT's code
    """
    if backward:
        x = np.flipud(x)

    sigma2 = 100
    X = np.zeros((int(np.ceil(extM/HOP)),extK)) # sets up matrix for EDMD estimation
    #x = np.hstack([xTrain,xTest])
    

    ## of length at least extM + extK
    for kk in range(extK):
        start = -extK - extM + kk
        end = kk - extK
        X[:,kk] = x[start:end:HOP]


    A = X[:,1:]
    B = x[-extM::HOP, None]
    
    Y = np.hstack([A, B])
    
    [Xi,mu,phi_end] = approximate_koopman(X,Y,sigma2)

    # X = [x1 x2 x3 ... ]
    # Z = [x1' x2' x3'...]

    start_time_ext = time.time()
    Z = np.zeros((round(np.ceil(extM/HOP)),L), dtype="complex_")
    tmp = phi_end.T
    #print(tmp)
    for kk in range(L):
        tmp = mu * tmp
        Z[:,kk] = tmp

    P = np.real(Xi.T @ Z)
    xext = P[-1,:].T
    print(f"EXTENTION TIME TAKEN FOR {L} INCREMENTS: {time.time() - start_time_ext}")

    if backward:
        xext = np.flipud(xext)
    return xext

def online_forecast(x,L,HOP,extK,extM, test):
    X = np.zeros((int(np.ceil(extM/HOP)),extK)) # sets up matrix for EDMD estimation
    #x = np.hstack([xTrain,xTest])
    

    ## of length at least extM + extK
    for kk in range(extK):
        start = -extK - extM + kk
        end = kk - extK
        X[:,kk] = x[start:end:HOP]


    A = X[:,1:]
    B = x[-extM::HOP, None]
    
    Y = np.hstack([A, B])

    redmd = RobustEDMD(extM)

    for i in range(1000):
        start_time = time.time()
        redmd.update_koopman_and_forecast_point(X[:, [i]], Y[:, [i]])
        print(f"ONLINE ITERATION TIME TAKEN: {time.time() -  start_time}")
        print(f"{redmd.timestep} {redmd.ready}")

    print(redmd.forecasts)

    lists = sorted(redmd.forecasts.items()) # sorted by key, return a list of tuples

    a, b = zip(*lists) # unpack a list of pairs into two tuples

    plt.figure(1)
    plt.subplot(211)
    plt.plot(b)
    plt.subplot(212)
    plt.plot(test)
    plt.show()


if __name__ == "__main__":

    
    timeEEG,sigEEG,fsEEG = loadData("F4M1.csv")
    timePleth,sigPleth,fsPleth = loadData("Pleth.csv")
    
    HOP = 1
    extSEC = 3 # extension length in seconds
    assert fsPleth == fsEEG
    L = round(extSEC * fsEEG)
    extM = round(1.5 * L)
    extK = round( 2.5 * extM ) # number of points to estimate A / size of datasets
    extKSecs = (extK/fsEEG )
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

        timeT = np.linspace(0,extKSecs + extMSecs,int((extKSecs + extMSecs)*fsEEG))
        timeTe = np.linspace(extKSecs + extMSecs,extKSecs + extMSecs + extSEC,int((extSEC)*fsEEG))
        start_time = time.time()
        xExt = forecast(seg['train'],L,HOP,extK,extM)
        print(f"FORECASTING TIME TAKEN: {time.time() -  start_time}")
        ax = plt.gca()
        plt.plot(timeT,seg['train'])
        plt.plot(timeTe,seg['test'],color='r')
        plt.plot(timeTe,xExt,color='b')
        plt.show()
        assert False
        metrics = calcMetrics(xExt,seg['test']) 
        eegStats.append(metrics)

        print(f"mse {metrics[0]}, l2 {metrics[1]}, linf {metrics[2]}")
        #print(seg['train'])

        plt.figure(1)
        plt.subplot(211)
        plt.plot(xExt)
        plt.subplot(212)
        plt.plot(seg['test'])
        plt.show()


        break
    
    '''
    plotMetrics(eegStats)
    for seg in plethData:

        xExt = forecast(seg['train'],L,HOP,extK,extM)

        metrics = calcMetrics(xExt,seg['test']) 
        plethStats.append(metrics)
    plotMetrics(plethStats)
    

    #for seg in eegData:
    #    online_forecast(seg['train'],L,HOP,extK,100, seg['test'])
    #    break

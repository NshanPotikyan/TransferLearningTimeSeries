import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def preprocessing(series):
    '''
    MinMax Scaling of the raw time series
    Args:
        series: the raw time series
    Returns:
        scaled_series and scaler object
    '''
    series = np.array(series)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1,1))
    scaled_series = scaled.reshape((len(series),))
    
    return scaled_series, scaler

def inverse_transform(series,scaler):
    '''
    Inverse transform of scales series
    Args:
        series: scaled series
        scaler: scaler object
    Returns:
        unscaled series
    '''
    return scaler.inverse_transform(series.reshape(-1,1))

def getSeries(data,p):
    '''
    Splits a given time series proportionally
    for training and testing purposes
    
    Args:
        data: numpy array or pandas series 
              containing the time series.
        p: float value that defines the 
           proportion of the series used
           for training.

    Returns:
        series: time series for training
        y_test: time series for testing
        n_test: number of timesteps 
                in the test series
    
    '''
    n = data.shape[0]
    n_train = int(n * p) 
    n_test = n - n_train

    x = np.arange(n)
        
    index_train = x[:n_train]
    index_test = x[n_train:]
    
    series = data[index_train]

    y_test = data[index_test]
    return series, y_test, n_test

def getInputOutput(series, input_size):
    '''
    Transforms the time series into desired 
    shape to be able to pass to the network
    
    Args:
        series: the time series.
        input_size: int that defines the length 
                    of the input sequence to be 
                    fed to the network
    Returns:
        X_train: input dataset
        y_train: output values
        X_test: the last available sequence
    
    '''
    
    series = np.array(series)
    xlen = len(series)
    xrows = xlen - input_size
    
    X_train, y_train = [], []
    
    for i in range(xrows):
        j = i + input_size
        a = series[i:j, np.newaxis]
        X_train.append(a)
        y_train.append(series[j])
    
    X_train,y_train = np.array(X_train), np.array(y_train)
    X_test = series[xrows:].reshape(1,input_size,1)
    
    
    return X_train, y_train, X_test

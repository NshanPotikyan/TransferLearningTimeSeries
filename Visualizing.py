import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def ACF(series,lags):
    '''
    Plots the Autocorrelation graph
    of the series
    Args:
        series: time series of interest
        lags: int specifying number of 
              periods to look back
    Returns: plot
    '''
    plot_acf(series, lags=lags)
    plt.savefig('ACF.png')
    plt.show()

def plotSeries(series_dict):
    '''
    Plots several time series
    Args:
        series_dict: dict with series names as keys
                     and time series as values
    Returns:
        plot
        
    '''
    n_series = len(series_dict) - 1
    names = list(series_dict.keys())
    
    plt.figure(figsize=(10,20))
    
    for i in range(n_series):
        plt.subplot(n_series,1,i+1)
        plt.title(names[i+1])
        plt.plot(series_dict['t'],series_dict[names[i+1]])
        
    plt.show()

def ViewLoss(history):
    '''
    Plots the history of model training
    '''
    plt.plot(history.history['loss'],label='Train')
    plt.plot(history.history['val_loss'],label='Val')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('loss-history.png')
    plt.show()

def view_predictions(series,predictions,actual,title):
    '''
    Plots the results of the predictions made by the model
    
    Args:
        series: the time series used for training the network
        predictions: numpy array containing the predicted values
        actual: the actual time series not seen by the network
        title: the title of the plot
        
    Returns:
        plot        
        
    '''  
    
    plt.figure(figsize=(8,4))
    plt.title(title)
    
    if isinstance(series,list):
        train_index = np.arange(len(series[0]))
        test_index = len(series[0]) + np.arange(len(actual))
        
        plt.plot(train_index,series[0], label = 'general')
        
    else:
        train_index = np.arange(len(series))
        test_index = len(series) + np.arange(len(actual))        
        plt.plot(train_index,series,label = 'training')

    if len(predictions) > 4:
        plt.plot(test_index,predictions,label = 'prediction',color='g')
        plt.plot(test_index,actual,label = 'actual',color='orange')
    else:
        plt.scatter(test_index,predictions,label = 'prediction',color='g')
        plt.scatter(test_index,actual,label = 'actual',color='orange')    
    
    plt.xlabel('Index')
    plt.ylabel('Data')
    
    plt.legend(loc='upper left')
    plt.savefig('{}_{}.png'.format(title,len(series)))
    plt.show()

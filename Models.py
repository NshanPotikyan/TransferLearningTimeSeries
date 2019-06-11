import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import itertools

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout




def build_LSTM(input_size, hidden_units, dropout, learning_rate):
    '''
    Builds the Network with LSTM hidden layers
    
    Args:
        input_size: int that defines the length 
                    of the input sequence to be 
                    fed to the network
        hidden_units: int/list specifying the number 
                      of hidden units in the hidden 
                      layer/layers
        dropout: boolean specifing whether to add dropout
                 with 0.5 rate per layer
        learning_rate: learning rate of the Adam 
                       optimization algorithm
    Returns:
        model: keras sequential model
        
    '''
    h = hidden_units
    

    model = Sequential()
    
    if isinstance(h,list):
    
        model.add(LSTM(h[0], 
                   batch_input_shape=(1,input_size, 1), 
                   return_sequences=True, 
                   stateful=True))
                  
        if dropout:
            model.add(Dropout(rate=0.5))

        if len(h) > 2:
            #removing 1st and last units
            for index, units in enumerate(h[1:-1]):  
                model.add(LSTM(units, 
                               batch_input_shape=(1,h[index], 1), 
                               return_sequences=True, 
                               stateful=True)) 
                if dropout:
                    model.add(Dropout(rate=0.5))

        model.add(LSTM(h[-1], 
                       batch_input_shape=(1,h[-2], 1), 
                       return_sequences=False, 
                       stateful=True))
        if dropout:
            model.add(Dropout(rate=0.5))
    else:
        model.add(LSTM(h, 
                   batch_input_shape=(1,input_size, 1), 
                   return_sequences=False, 
                   stateful=True)) 
        if dropout:
            model.add(Dropout(rate=0.5))
        
    
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    return model

def predict_ahead(model,X_test,n_ahead):
    '''
    Makes predictions based on the last available sequence
    
    Args:
        model: keras sequential model
        X_test: the last available sequence
        n_ahead: number of predictions to make 
        
    Returns:
        predictions: numpy array containing the predicted values
        
    '''    
    predictions = np.zeros(n_ahead)
    predictions[0] = model.predict(X_test,batch_size = 1)
    
    if n_ahead > 1:
        for i in range(1,n_ahead):
            x_new = np.append(X_test[0][1:],predictions[i-1])
            X_test = x_new.reshape(1,x_new.shape[0],1)
            predictions[i] = model.predict(X_test,batch_size = 1)
    return predictions

def FitForecast(X_train, y_train, X_test, n_ahead, input_size,
                hidden_units, dropout, val_split, learning_rate, 
                epochs, trained_model):
    
    '''
    Fits a model and returns the predicted values. 
    Optionally weights from another network can be passed 
    
    Args:
        X_train: input dataset for training
        y_train: output dataset for training
        X_test: the last available sequence
        n_ahead: number of predictions to make 
        time_series: the time series of interest
        input_size: int that defines the length 
                    of the input sequence to be 
                    fed to the network
        hidden_units: int/list specifying the number 
                      of hidden units in the hidden 
                      layer/layers
        dropout: boolean specifing whether to add dropout
                 with 0.5 rate per layer
        learning_rate: learning rate of the Adam 
                       optimization algorithm
        epochs: int that defines the number of 
                training phases through the
                training dataset
        trained_model: already trained keras sequential 
                       model
        
    Returns:
        model: keras sequential model
        predictions: numpy array containing the predicted values
        history: training and validation loss history
        
    '''
    model = build_LSTM(input_size,hidden_units,dropout, learning_rate)
    
    if trained_model is not None:
        model.set_weights(weights = trained_model.get_weights())        
    
    
    history = model.fit(x=X_train, y=y_train, 
                batch_size=1, epochs=epochs, 
                verbose=1, validation_split=val_split,
                shuffle=False)

    predictions = predict_ahead(model,X_test,n_ahead)
    return model, predictions, history

def FitEvaluate(time_series,params):
    '''
    Calls the pipeline to fit an LSTM model to the 
    given time series
    
    Args:
        time_series: the time series of interest
        params: a dictionary specifying parameters
                {input_size, hidden_units, dropout,
                learning_rate, n_ahead, val_split, 
                epochs, verbose, plot}
    Returns:
        model: keras sequential model      
        mse: mean squared error of the prediction
        history: training and validation loss history
        
    '''   
    
    for k in params.keys():
        globals()[k] = params[k]
    
    
    scaled_series, scaler = preprocessing(time_series)
    series, y_test, n_test = getSeries(scaled_series,0.8)
    X_train,y_train,X_test = getInputOutput(series,input_size)
    
    # show only n_ahead number of actual values
    y_test = y_test[np.arange(n_ahead)]

    new_model, predictions, history = FitForecast(X_train,y_train,X_test,n_ahead,
                                        input_size,hidden_units,dropout, val_split,
                                        learning_rate,epochs,trained_model=None)
    
    # rescaling
    series = inverse_transform(series, scaler)
    y_test = inverse_transform(y_test, scaler)
    predictions = inverse_transform(predictions, scaler)
    
    mse = mean_squared_error(y_true=y_test,y_pred=predictions)
    
    if verbose:
        print('\n')
        print('======== Prediction Evaluation =========')
        print('MSE is {}'.format(round(mse,4)))
        
    if plot:
        ViewLoss(history)
        view_predictions(series,predictions,y_test,'Actual vs Forecast')
    return new_model, mse, history

def TransferLearning(time_series,params,model):
    '''
    Calls the pipeline to fit an LSTM model to the 
    given time series with and without knowledge
    transfer
    
    Args:
        time_series: the time series of interest
        params: a dictionary specifying parameters
                {input_size, hidden_units, dropout,
                learning_rate, n_ahead, val_split, 
                epochs, verbose, plot}
        model: already trained keras sequential 
               model
    Returns:
        mean squared errors of the predictions
        and 2 plots         
    '''
    for k in params.keys():
        globals()[k] = params[k]
    
    val_split = 0
    
    scaled_series, scaler = preprocessing(time_series)
    series, y_test, n_test = getSeries(scaled_series,0.8)
    X_train,y_train,X_test = getInputOutput(series,input_size)

    # show only n_ahead number of actual values
    y_test = y_test[np.arange(n_ahead)]

    
    print('*** Fitting a model without knowledge transfer ***')
    model_noTransfer, predictions_noTransfer, _ = FitForecast(X_train,y_train,
                                                             X_test,n_ahead,
                                               input_size,hidden_units,
                                                             dropout,val_split,
                                                             learning_rate,
                                               epochs,
                                                             trained_model=None)
    
    print('\n')
    print('*** Fitting a model with knowledge transfer ***')
    model_withTransfer, predictions_withTransfer, _ = FitForecast(X_train,y_train,
                                                                 X_test,n_ahead,
                                                                 input_size,hidden_units,
                                                                 dropout,val_split,
                                                                 learning_rate,
                                                                 epochs,
                                                                 trained_model=model)
    
    # rescaling
    series = inverse_transform(series, scaler)
    y_test = inverse_transform(y_test, scaler)
    predictions_noTransfer = inverse_transform(predictions_noTransfer, scaler)
    predictions_withTransfer = inverse_transform(predictions_withTransfer, scaler)
    
    mse_noTransfer = mean_squared_error(y_true=y_test,y_pred=predictions_noTransfer)
    mse_withTransfer = mean_squared_error(y_true=y_test,y_pred=predictions_withTransfer)
      
    print('\n')
    print('======== Results for no knowledge transfer =========')
    print('The RMSE is {}'.format(round(np.sqrt(mse_noTransfer),4)))
    print('\n')
    print('======== Results for knowledge transfer =========')
    print('The RMSE is {}'.format(round(np.sqrt(mse_withTransfer),4)))
        
    view_predictions(series,predictions_noTransfer,y_test,title='Without Transfer')
    view_predictions(series,predictions_withTransfer,y_test,title='With Transfer')

def GridSearch(series,params_grid):
    '''
    Runs a grid search over specified parameter ranges
    Args: 
        series: the time series of interest
        params_grid: a dictionary specifying parameters
                    {input_size, hidden_units, dropout,
                    learning_rate, n_ahead, val_split, 
                    epochs, verbose, plot} and their 
                    possible value ranges
    Returns:
        model: the model with the lowest MSE 
        logs: logs of all combinations        
       
    '''
    param_names = list(params_grid.keys())
    param_values = list(params_grid.values()) 
    combinations = list(itertools.product(*param_values))
    
    logs = pd.DataFrame(combinations,columns=param_names)
    
    mse_prev = 1
    for index, comb in enumerate(combinations):
        
        print('Fitting {}/{} model'.format(index+1,len(combinations)))
        params = dict(zip(param_names,comb))
        model, mse, history = FitEvaluate(series,params)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        
        if mse < mse_prev:
            mse_prev = mse
            best_model = model
        
        logs.at[index,'mse'] = mse
        logs.at[index,'mean_training_loss'] = np.mean(train_loss)
        logs.at[index,'std_training_loss'] = np.std(train_loss)
        logs.at[index,'mean_val_loss'] = np.mean(val_loss)
        logs.at[index,'std_val_loss'] = np.std(val_loss)
    logs.to_csv('results.csv',index=False)
    view_predictions(series,best_predictions,y_test,'Actual vs Forecast')
    return best_model, logs


def generalTuning(series1, series2, params):
    
    '''
    Fitting 3 models trained on 
    {general domain,general+in-domain,in-domain only}
    and comparing them
    Args: 
        series1: target related time series 
        series2: target time series
        params: a dictionary specifying parameters
                {input_size, hidden_units, dropout,
                learning_rate, n_ahead, val_split, 
                epochs, verbose, plot}
    Returns:
        mean squared errors of the predictions
        and 3 plots 
    
    '''
    
    time_series = np.concatenate([series1,series2])
        
    
    
    for k in params.keys():
        globals()[k] = params[k]
        
    val_split = 0
    
    # preprocessing general series
    scaled_general, scaler_general = preprocessing(time_series)
    series_general, _, __ = getSeries(scaled_general,0.9)
    X_train_general,y_train_general,___ = getInputOutput(series_general,input_size)

    # preprocessing the target series
    scaled_target, scaler_target = preprocessing(series2) 
    series_target, y_test_target, n_test = getSeries(scaled_target,0.8)
    X_train_target,y_train_target,X_test_target = getInputOutput(series_target,input_size)

    # comparing predictions with only n_ahead number of actual values
    y_test_target = y_test_target[np.arange(n_ahead)]

    
    # build and train a model on the general domain (time_series)
    
    print('*** Fitting a model on general domain ***')

    
    model_general, predictions_pre_tuned, hist = FitForecast(X_train_general,y_train_general,
                                                             X_test_target,n_ahead,
                                                             input_size, hidden_units,
                                                             dropout, val_split,
                                                             learning_rate,
                                                             epochs,
                                                             trained_model=None)                                               
                                                             
                                                             
                                                        
 
    # initiallize a model for target
    model_tuned = build_LSTM(input_size, hidden_units, dropout, learning_rate)
        
    # transfer the knowledge from the pre-trained model
    # and tune it only on the target domain (series2)
    model_tuned.set_weights(weights=model_general.get_weights())        
    
    
    print('\n *** Tuning a model on target domain ***')

    model_tuned.fit(x=X_train_target, y=y_train_target, 
                        batch_size=1, epochs=epochs, 
                        verbose=1, validation_data=None,
                        shuffle=False)
    
    predictions_tuned = predict_ahead(model_tuned,X_test_target,n_ahead)
    
    
   
    print('\n *** Fitting a model on target domain only ***')

    model_target, predictions_target,hist2 = FitForecast(X_train_target,y_train_target,
                                                   X_test_target,n_ahead,
                                                   input_size, hidden_units,
                                                   dropout, val_split,
                                                   learning_rate,
                                                   2 * epochs,
                                                   trained_model=None)                                               


    series_target = inverse_transform(series_target,scaler_target)
    y_test_target = inverse_transform(y_test_target,scaler_target)
    
    series_general = inverse_transform(series_general,scaler_general)
    predictions_pre_tuned = inverse_transform(predictions_pre_tuned,scaler_target)
    mse_pre_tuned = mean_squared_error(y_true=y_test_target,y_pred=predictions_pre_tuned)
    
    predictions_tuned = inverse_transform(predictions_tuned,scaler_target)
    mse_tuned = mean_squared_error(y_true=y_test_target,y_pred=predictions_tuned)
    
    predictions_target = inverse_transform(predictions_target,scaler_target)  
    mse_target = mean_squared_error(y_true=y_test_target,y_pred=predictions_target)
    
    print('\n')
    print('======== Results for pre_tuned model =========')
    print('The RMSE is {}'.format(round(np.sqrt(mse_pre_tuned),4)))
    print('\n')
    print('======== Results for tuned model =========')
    print('The RMSE is {}'.format(round(np.sqrt(mse_tuned),4)))
    
    print('\n')
    print('======== Results for target model only =========')
    print('The RMSE is {}'.format(round(np.sqrt(mse_target),4)))
    
        
    view_predictions([series_general,series_target],predictions_pre_tuned,y_test_target,title='Pre-tuned')
    view_predictions([series_general,series_target],predictions_tuned,y_test_target,title='Tuned')
    view_predictions(series_target,predictions_target,y_test_target,title='Target only')



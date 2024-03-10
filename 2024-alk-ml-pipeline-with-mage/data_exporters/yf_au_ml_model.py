import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from sklearn.metrics import root_mean_squared_error

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(df, *args, **kwargs):
    #create feature matrix X
    X = df[['Adj Close_Gold', 'log_returns']].values

    #data normalization with MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_scaled = scaler.transform(X)

    #create target vector y
    y = [x[0] for x in X_scaled]

    #data split into train and test
    split_1 = int(len(X_scaled) * 0.7)
    split_2 = int(len(X_scaled) * 0.9)

    X_train = X_scaled[:split_1]
    X_validation = X_scaled[split_1 : split_2]
    X_test = X_scaled[split_2 : len(X_scaled)]
    y_train = y[:split_1]
    y_validation = y[split_1 : split_2]
    y_test = y[split_2 : len(y)]

    ####################################################
    #test the lengths
    #assert len(X_train) == len(y_train)
    #assert len(X_validation) == len(y_validation)
    #assert len(X_test) == len(y_test)
    ####################################################

    ###labeling

    n = 3
    Xtrain = []
    ytrain = []
    Xvalidation = []
    yvalidation = []
    Xtest = []
    ytest = []
    for i in range(n, len(X_train)):
        Xtrain.append(X_train[i - n: i, : X_train.shape[1]])
        ytrain.append(y_train[i])
    for i in range(n,len(X_validation)):
        Xvalidation.append(X_validation[i - n: i, : X_validation.shape[1]])
        yvalidation.append(y_validation[i])
    for i in range(n, len(X_test)):
        Xtest.append(X_test[i - n: i, : X_test.shape[1]])
        ytest.append(y_test[i])
        
    #LSTM inputs reshaping
    Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

    Xvalidation, yvalidation = (np.array(Xvalidation), np.array(yvalidation))
    Xvalidation = np.reshape(Xvalidation, (Xvalidation.shape[0], Xvalidation.shape[1], Xvalidation.shape[2]))

    Xtest, ytest = (np.array(Xtest), np.array(ytest))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))

    #LSTM model
    model = Sequential()
    #1st LSTM layer with 5 neurons, dropout 20%
    #dropout - regularization technique to prevent overfitting
    #by randomly dropping a fraction of neurons during training (20% of neurons will be set to zero)
    model.add(LSTM(5, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), dropout=0.2, return_sequences=True))
    #2nd LSTM layer with 10 neurons, dropout 20%
    model.add(LSTM(10, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), dropout=0.2))
    #1st dense layer with 5 neurons
    #relu - Rectified Linear Activation, it replaces all negative values in the input tensor with zero
    #and maintains all positive values
    model.add(Dense(5, activation='relu'))
    #output dense layer with 1 neuron
    model.add(Dense(1))
    #loss function specifies the function to minimize
    #optimizer specifies the algorithm to use to minimize the loss function
    #adam - Adaptive Moment Estimation, it is a method for stochastic optimization
    #which dinamically updates the learning rate
    model.compile(loss='mean_squared_error', optimizer='adam')
    #verbose - amount of info printed during the training, verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
    model.fit(
        Xtrain, ytrain, epochs=10, validation_data=(Xvalidation, yvalidation), batch_size=32, verbose=1
    )

    model.summary()

    train_predict = model.predict(Xtrain)
    validation_predict = model.predict(Xvalidation)
    test_predict = model.predict(Xtest)

    train_predict = np.c_[train_predict, np.zeros(train_predict.shape)]
    validation_predict = np.c_[validation_predict, np.zeros(validation_predict.shape)]
    test_predict = np.c_[test_predict, np.zeros(test_predict.shape)]

    #invert prediction
    train_predict = scaler.inverse_transform(train_predict)
    train_predict = [x[0] for x in train_predict]

    validation_predict = scaler.inverse_transform(validation_predict)
    validation_predict = [x[0] for x in validation_predict]

    test_predict = scaler.inverse_transform(test_predict)
    test_predict = [x[0] for x in test_predict]

    #calculate square root of mean squared error
    train_score = root_mean_squared_error([x[0][0] for x in Xtrain], train_predict)
    print('Train Score: %.2f RMSE' % (train_score))

    validation_score = root_mean_squared_error([x[0][0] for x in Xvalidation], validation_predict)
    print('Validation Score: %.2f RMSE' % (validation_score))

    test_score = root_mean_squared_error([x[0][0] for x in Xtest], test_predict)
    print('Test Score: %.2f RMSE' % (test_score))

    #results visualization
    #data split into train and test
    df_train = df[n:split_1]
    df_validation = df[split_1+n : split_2]
    df_test = df[split_2+n : len(X_scaled)]

    df_test['predictions'] = test_predict

    #plot actual vs predicted values
    #plt.plot(df_train.iloc[-(int(0.25*len(df_train))):]['Adj Close_Gold'], label='25% of Train')
    #plt.plot(df_validation['Adj Close_Gold'], label='Validation')
    plt.plot(df_test['Adj Close_Gold'], label='Actual')
    plt.plot(df_test['predictions'], label='Forecast')
    plt.legend(loc='upper left')
    plt.xticks(rotation=90)
    plt.show();

@test
def test_output(output, *args) -> None:
    #test the lengths
    assert len(X_train) == len(y_train)
    assert len(X_validation) == len(y_validation)
    assert len(X_test) == len(y_test)

    assert output is not None, 'The output is undefined'
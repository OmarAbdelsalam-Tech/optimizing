from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, LSTM, Bidirectional, concatenate, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/abdel/Desktop/Projects/General Attack -Version 0/Badr Data/training_features.csv")
labels = pd.read_csv("C:/Users/abdel/Desktop/Projects/General Attack -Version 0/Badr Data/training_labels.csv")

# Split the features and target into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2)


def create_model(num_filters, kernel_size, num_lstm_units, num_dense_units, dropout_rate, lr):
    inputs = Input(shape=(x_train.shape[1], 1))
    num_classes = 1

    conv1 = Conv1D(num_filters, kernel_size, padding='same', activation='relu')(inputs)
    lstm1 = Bidirectional(LSTM(num_lstm_units, return_sequences=True))(conv1)
    lstm2 = Bidirectional(LSTM(num_lstm_units, return_sequences=True))(lstm1)
    concat = concatenate([lstm1, lstm2])
    flatten = Flatten()(concat)
    dense1 = Dense(num_dense_units, activation='relu')(flatten)
    dropout = Dropout(dropout_rate)(dense1)
    dense2 = Dense(num_classes, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=dense2)

    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameter space
param_dist = {'num_filters': sp_randint(16, 512),
              'kernel_size': sp_randint(2, 10),
              'num_lstm_units': sp_randint(16, 512),
              'num_dense_units': sp_randint(64, 512),
              'dropout_rate': np.arange(0, 0.9, 0.1),
              'lr': [0.001, 0.01, 0.1]}

# Define search parameters
n_iter_search = 100
cv = 30
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter_search, cv=cv, verbose=2)

# Perform hyperparameter search
random_search.fit(x_train, y_train)

# Print results
print('Best parameters:', random_search.best_params_)
print('Best score:', random_search.best_score_)

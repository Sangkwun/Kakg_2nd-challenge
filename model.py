import os
import subprocess
import tensorflow as tf
import numpy as np
import pandas as pd

from utils import root_mean_squared_error
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K


class Model():
    '''
        TODO:
        - k-fold 다른 방식으로 구현하기.
        - 
    '''
    def __init__(self, x_train, x_test, y_train, y_scaler, output='output.csv'):
        self.output = output
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train

        self.y_scaler = y_scaler

        self.model_path = './model/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path1 = os.path.join(self.model_path, 'best_model1.hdf5')
        self.model_path2 = os.path.join(self.model_path, 'best_model2.hdf5')

        self.epoch = 150
        self.patient = 20
        self.k = 4
        self.num_val_samples = len(self.x_train) // self.k

        self.callbacks1 = [
            EarlyStopping(monitor='val_loss', patience=self.patient, mode='min', verbose=1),
            ModelCheckpoint(filepath=self.model_path1, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = self.patient / 3, min_lr=0.000001, verbose=1, mode='min')
        ]

        self.callbacks2 = [
            EarlyStopping(monitor='val_loss', patience=self.patient, mode='min', verbose=1),
            ModelCheckpoint(filepath=self.model_path2, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = self.patient / 3, min_lr=0.000001, verbose=1, mode='min')
        ]

    def make_model1(self):
        model = models.Sequential()
        model.add(layers.Dense(8, activation='relu',	input_dim=self.x_train.shape[1]))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(1))

        optimizer = optimizers.Adam(lr=0.001)
        model.compile(optimizer= optimizer, loss = root_mean_squared_error, metrics=[root_mean_squared_error, 'mse', 'mae'])
        return model

    def make_model2(self):
        model = models.Sequential()
        model.add(layers.Dense(8, activation='relu',	input_dim=self.x_train.shape[1]))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(1))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(optimizer= optimizer, loss = root_mean_squared_error, metrics=[root_mean_squared_error, 'mse', 'mae'])
        return model
    
    def start(self):
        for i in range(self.k):
            print('Fold num #', i+1)
            val_data = self.x_train[i * self.num_val_samples : (i+1) * self.num_val_samples]
            val_targets = self.y_train[i * self.num_val_samples : (i+1) * self.num_val_samples]

            partial_train_data = np.concatenate(
                [self.x_train[:i*self.num_val_samples],
                self.x_train[(i+1) * self.num_val_samples:]],
                axis=0
            )
            partial_train_targets = np.concatenate(
                [self.y_train[:i*self.num_val_samples],
                self.y_train[(i+1) * self.num_val_samples:]],
                axis=0
            )
            
            model1 = self.make_model1()
            model2 = self.make_model2()
            
            history1 = model1.fit(
                partial_train_data, 
                partial_train_targets,
                validation_data=(val_data, val_targets), 
                epochs=self.epoch, 
                batch_size=16, 
                callbacks=self.callbacks1)

            history2 = model2.fit(
                partial_train_data, 
                partial_train_targets,
                validation_data=(val_data, val_targets), 
                epochs=self.epoch, 
                batch_size=16, 
                callbacks=self.callbacks2)

        best_model1 = self.make_model1()
        best_model1.load_weights('./model/best_model1.hdf5')
        best_model1.fit(
            self.x_train, 
            self.y_train, 
            epochs=self.epoch, 
            batch_size=8, 
            shuffle=True, 
            validation_split=0.2,
            callbacks=self.callbacks1
            )

        best_model2 = self.make_model2()
        best_model2.load_weights('./model/best_model2.hdf5')
        best_model2.fit(
            self.x_train, 
            self.y_train, 
            epochs=self.epoch, 
            batch_size=8, 
            shuffle=True, 
            validation_split=0.2,
            callbacks=self.callbacks2
            )

        y_preds_best1 = best_model1.predict(self.x_test)
        inv_y_preds_best1 =  np.expm1(self.y_scaler.inverse_transform(y_preds_best1))

        y_preds_best2 = best_model2.predict(self.x_test)
        inv_y_preds_best2 =  np.expm1(self.y_scaler.inverse_transform(y_preds_best2))

        avg_pred = ( inv_y_preds_best1 + inv_y_preds_best2 + inv_y_preds_best3 - 3) / 3
        avg_pred = avg_pred.astype(int)

        self.result = avg_pred
        self.save()
    
    def save(self):
        sub = pd.read_csv('./dataset/sample_submission.csv')
        sub['price'] = self.result
        sub.to_csv(self.output, index=False)

    def submit(self, message='submit'):
        command = 'kaggle competitions submit -c 2019-2nd-ml-month-with-kakr -f {} -m {}'.format(self.output, message)
        subprocess.call (command, shell=True)
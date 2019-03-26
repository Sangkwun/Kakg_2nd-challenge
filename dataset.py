import pandas as pd
import numpy as np

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from utils import deleteTextFromColumn
from utils import splitTextDate


class Dataset():
    '''
        TODO:
        - valid, train split하기
        - k-fold 여기해서 해서 보내기
        - outlier 제거하기
    '''
    def __init__(self):
        self.df_train = pd.read_csv('./dataset/train.csv')
        self.df_test = pd.read_csv('./dataset/test.csv')

        self.process_text()
        self.process_long()
        self.check_attic()
        self.check_basement()
        self.check_living_increased()
        self.check_lot_increased()
        self.total_sqrt()
        self.total_sqrt15()
        self.check_total_increased()
        self.apply_log()

    def process_text(self):
        self.df_train = deleteTextFromColumn(self.df_train, 'date', 'T000000')
        self.df_test = deleteTextFromColumn(self.df_test, 'date', 'T000000')

        self.df_train = splitTextDate(self.df_train)
        self.df_test = splitTextDate(self.df_test)

    def process_long(self):
        self.df_train['long'] = np.abs(self.df_train['long'])
        self.df_test['long'] = np.abs(self.df_test['long'])

    def check_attic(self):
        self.df_train['attic'] = self.df_train['floors'] % 1 == 0.5
        self.df_test['attic'] = self.df_test['floors'] % 1 == 0.5

        self.df_train['floors'] = self.df_train['floors'] // 1
        self.df_train['floors'] = self.df_train['floors'] // 1

    def check_basement(self):
        self.df_train['has_basement'] = self.df_train['sqft_basement'] > 0
        self.df_test['has_basement'] = self.df_test['sqft_basement'] > 0
    
    def check_living_increased(self):
        self.df_train['living_increased'] = self.df_train['sqft_living15'] > self.df_train['sqft_living']
        self.df_test['living_increased'] = self.df_test['sqft_living15'] > self.df_test['sqft_living']
    
    def check_lot_increased(self):
        self.df_train['living_increased'] = self.df_train['sqft_lot15'] > self.df_train['sqft_lot']
        self.df_test['living_increased'] = self.df_test['sqft_lot15'] > self.df_test['sqft_lot']

    def total_sqrt(self):
        self.df_train['total_sqrt'] = self.df_train['sqft_living'] + self.df_train['sqft_lot']
        self.df_test['total_sqrt'] = self.df_test['sqft_living'] + self.df_test['sqft_lot']
    
    def total_sqrt15(self):
        self.df_train['total_sqrt15'] = self.df_train['sqft_living15'] + self.df_train['sqft_lot15']
        self.df_test['total_sqrt15'] = self.df_test['sqft_living15'] + self.df_test['sqft_lot15']

    def check_total_increased(self):
        self.df_train['total_increased'] = self.df_train['total_sqrt15'] > self.df_train['total_sqrt']
        self.df_test['total_increased'] = self.df_test['total_sqrt15'] > self.df_test['total_sqrt']

    def apply_log(self):
        log_list = ['price', 'long', 'lat', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above',
                 'sqft_basement', 'yr_built', 'sqft_living15', 'sqft_lot15', 'total_sqrt', 'total_sqrt15']
        for name in log_list:
            self.df_train[name] = np.log(self.df_train[name] + 1)
            if name is not 'price':
                self.df_test[name] = np.log(self.df_test[name] + 1) # df_test doesn't contain `price`

    def get_train_testset(self):
        df_train_dummy = pd.get_dummies(self.df_train.drop(['date'], axis=1))
        df_test_dummy = pd.get_dummies(self.df_test.drop(['date'], axis=1))

        x_train = df_train_dummy.drop(['price', 'id'], axis=1)
        x_test = df_test_dummy.drop(['id'], axis=1)
        y_train = df_train_dummy['price']
        y_train = y_train.values.reshape(-1,1)

        x_scaler = StandardScaler().fit(x_train)
        y_scaler = StandardScaler().fit(y_train)

        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)
        y_train = y_scaler.transform(y_train)

        self.scaler = y_scaler
        return x_train, x_test, y_train
    
    def get_scaler(self):
        return self.scaler


if __name__ == "__main__":
    dataset = Dataset()
    x_train, x_test, y_train = dataset.get_train_testset()
    print(x_train.shape)
    print(x_train[0])
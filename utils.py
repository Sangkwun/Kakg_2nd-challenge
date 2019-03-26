
import pandas as pd
from keras import backend as k

def deleteTextFromColumn(df, column_name, text):
    df[column_name] = df[column_name].str.replace(text, '')
    return df

def splitTextDate(df):
    if 'date' not in df.columns:
        text = 'No date column is in dataframe'
        raise ValueError(text)
    
    df['year'] = pd.to_numeric(df['date'].str[0:4])
    df['month'] = pd.to_numeric(df['date'].str[4:6])
    df['day'] = pd.to_numeric(df['date'].str[6:])
    
    return df

def root_mean_squared_error(y_true, y_pred):
    return k.sqrt(k.mean(k.square(y_pred - y_true), axis=-1)) 
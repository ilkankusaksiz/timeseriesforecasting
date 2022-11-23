
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('C://Users//leptop//Desktop//municipality_bus_utilization.csv')
# i checked my date columns type and that is string.
type(df.timestamp[0])

df = pd.read_csv('C://Users//leptop//Desktop//municipality_bus_utilization.csv',parse_dates=["timestamp"])
# when i check my date columns type one more time, it is timestamp now.
type(df.timestamp[0])

# i checked null values in my dataset and there is no null value.
df['total_capacity'].isnull().sum()
df['usage'].isnull().sum()
df['municipality_id'].isnull().sum()



# i classified my data by municipality_id
grp = df.groupby('municipality_id')
city_0 = grp.get_group(0)
city_1 = grp.get_group(1)
city_2 = grp.get_group(2)
city_3 = grp.get_group(3)
city_4 = grp.get_group(4)
city_5 = grp.get_group(5)
city_6 = grp.get_group(6)
city_7 = grp.get_group(7)
city_8 = grp.get_group(8)
city_9 = grp.get_group(9)

names= [city_0,city_1,city_2,city_3,city_4,city_5,city_6,city_7,city_8,city_9]

for city in names:
# After i seperated my data, there is wrong index number and i need to reset it. I removed city names from my dataset.
    city.reset_index(drop=True, inplace=True)
    city.drop(columns=['municipality_id'],inplace=True)
    
    from datetime import datetime, date 
    df = city
    


    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 25))

    sns.lineplot(x=df.timestamp, y=df.usage.fillna(np.inf), ax=ax[0], color='dodgerblue',label='People with time')
    ax[0].set_title('Feature: usage', fontsize=14)
    ax[0].set_ylabel(ylabel='usage', fontsize=14)


    sns.lineplot(x=df.timestamp, y=df.total_capacity.fillna(np.inf), ax=ax[1], color='dodgerblue',label='Total Capacity')
    ax[1].set_title('Feature: total_capacity', fontsize=14)
    ax[1].set_ylabel(ylabel='total_capacity', fontsize=14)
    
    df['usage'] = df['usage'].interpolate()
    df['total_capacity'] = df['total_capacity'].interpolate()
    
    
    df['year'] = pd.DatetimeIndex(df['timestamp']).year
    df['month'] = pd.DatetimeIndex(df['timestamp']).month
    df['day_of_week'] = pd.DatetimeIndex(df['timestamp']).dayofweek
    df['day'] = pd.DatetimeIndex(df['timestamp']).day
    df['hour'] = pd.DatetimeIndex(df['timestamp']).hour


    df[['year', 'month', 'day', 'day_of_week', 'hour']].head()

    from sklearn.model_selection import TimeSeriesSplit

    N_SPLITS = 2
    
    X = df.timestamp
    y = df.usage
    
    folds = TimeSeriesSplit(n_splits=N_SPLITS)
    
    #df = df.drop('timestamp', axis=1)
    
    
    target = 'usage'
    features = [feature for feature in df.columns if feature != target]
    
    split_date = '2017-08-05'
    df_train = df.loc[df['timestamp'] <=split_date ].copy()
    df_test = df.loc[df['timestamp']> split_date].copy()


    def create_features(df, label=None):
        """
        Creates time series features from datetime index
        """
        df['year'] = pd.DatetimeIndex(df['timestamp']).year
        df['month'] = pd.DatetimeIndex(df['timestamp']).month
        df['day'] = pd.DatetimeIndex(df['timestamp']).day
        df['hour'] = pd.DatetimeIndex(df['timestamp']).hour
        
        
        X = df[['hour','day','month','year']]
        if label:
            y = df[label]
            return X, y
        return X

    X_train, y_train = create_features(df_train, label='usage')
    X_test, y_test = create_features(df_test, label='usage')

    from xgboost import plot_importance, plot_tree
    reg = xgb.XGBRegressor(n_estimators=100)
    reg.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],early_stopping_rounds=50,verbose=False) # Change verbose to True if you want to see it train
    _ = plot_importance(reg, height=0.9)


    df_test['Prediction'] = reg.predict(X_test)
    df_all = pd.concat([df_test, df_train], sort=False)

    _ = df_all[['usage','Prediction']].plot(figsize=(15, 5))

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mean_squared_error(y_true=df_test['usage'],y_pred=df_test['Prediction'])
    mean_absolute_error(y_true=df_test['usage'],y_pred=df_test['Prediction'])
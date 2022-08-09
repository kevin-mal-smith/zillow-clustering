import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from env import host, username, password
import os
import numpy as np
from matplotlib import cm

from sklearn.model_selection import learning_curve

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

def get_zillow_data():
    
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv') 
      
    return df




def get_connection(db, user=username, host=host, password=password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
   
    sql_query = """select prop.parcelid
        , pred.logerror
        , bathroomcnt
        , bedroomcnt
        , calculatedfinishedsquarefeet
        , fips
        , latitude
        , longitude
        , lotsizesquarefeet
        , regionidcity
        , regionidcounty
        , regionidzip
        , yearbuilt
        , structuretaxvaluedollarcnt
        , taxvaluedollarcnt
        , landtaxvaluedollarcnt
        , taxamount
        from properties_2017 prop
    inner join predictions_2017 pred on prop.parcelid = pred.parcelid
    where propertylandusetypeid = 261;

                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df = df.drop_duplicates(subset=['parcelid'],keep = 'last')
    return df

def clean_data(df):
    df['age'] = 2017 - df.yearbuilt
    df['acres'] = df.lotsizesquarefeet/43560
    df['land_cost_per_sqf'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    df['structure_cost_per_sqf'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet
    df['tax_rate'] = df.taxamount/df.taxvaluedollarcnt
    df = df.drop(columns=['taxamount','taxvaluedollarcnt'])
    df['bed_bath'] = df.bedroomcnt/df.bathroomcnt
    df.columns = ['parcelid','logerror','bed','bath','sq_feet','latitude','longitude','lot_size','city_id','zip','year','home_value'
             ,'land_value','la','orange','ventura','age','acres','land_cost/sqf','home_cost/sqf','tax_rate','bed_bath_ratio']
    return df


def remove_outliers(df):
    return df[(((df.bath <= 7)&(df.bath > 0)) & ((df.bed <= 7)&(df.bed > 0)) & 
               (df.zip < 100000) & 
               (df.acres < 20) &
               (df.sq_feet < 10000) & 
               (df.tax_rate < 10)
              )]


def wrangle_zillow():
    df = get_zillow_data()

    df= df.dropna()

    counties = pd.get_dummies(df.fips)

    counties.columns = ['la','orange','ventura']
    df = pd.concat([df,counties],axis = 1)
    df = df.drop(columns=['fips','regionidcounty'])

    df = clean_data(df)
    
    df = remove_outliers(df)

    train, validate = train_test_split(df,random_state=123)
    train, test = train_test_split(train, random_state=123)

    return train, validate, test



def scale_and_concat(df):
    continuous=['sq_feet','latitude','longitude','home_value','land_value','acres','land_cost/sqf','home_cost/sqf','tax_rate']
    scaler = MinMaxScaler()
    scaler.fit(df[continuous])
    scaled_column_names = ['scaled_' + i for i in continuous]
    scaled_array = scaler.transform(df[continuous])
    scaled_df = pd.DataFrame(scaled_array, columns=scaled_column_names, index=df.index.values)
    return pd.concat((df, scaled_df), axis=1)

def prep(train,validate,test):
    train_scaled=scale_and_concat(train)
    validate_scaled = scale_and_concat(validate)
    test_scaled = scale_and_concat(test)

    continuous=['sq_feet','latitude','longitude','home_value','land_value','acres','land_cost/sqf','home_cost/sqf','tax_rate']

    train_scaled = train_scaled.drop(columns=continuous)
    validate_scaled = validate_scaled.drop(columns=continuous)
    test_scaled = test_scaled.drop(columns=continuous)
    return train_scaled, validate_scaled, test_scaled


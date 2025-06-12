#%%
#Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None) # configuration to visualize all the columns

#%%
# exraction
def extract(file):
    df = pd.read_csv(file)
    return df

def shape(df):
    print('Shape')
    print(df.shape)
    print(f"The number of rows is {df.shape[0]}, and the number of columns is {df.shape[1]}")
    print('---------------------')
    print('The columns are:')
    print(df.columns)

# Variables with nulls
def nulls(df,count=0,share=0):
    
    nulls = df.isnull().sum()
    nulls_share = df.isnull().sum()/df.shape[0]*100
    with_nulls_share = nulls_share[nulls_share > 0]
    
    if count == 1:
        print('Count of nulls')
        with_nulls = nulls[nulls > 0]
        print (with_nulls.sort_values(ascending=False))
        print('-----------------------------------')
    
    if share == 1:
        print('share of nulls')
        print (with_nulls_share.sort_values(ascending=False))
    
    nulls_list = with_nulls_share.to_frame(name='perc_nulos').reset_index().rename(columns={'index': 'var'})
    
    return nulls_list


#%%
file = 'spaces_hr_raw_data.csv'
df = extract(file)

shape(df)
nulls(df,1,1)

df = df.drop(columns=['numberchildren', 'over18', 'yearsincurrentrole'])
df.shape
# %%

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

# Transform
def transform(df):
    # Rename columns for consistency
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Drop redundant or irrelevant columns
    df = df.drop(columns=['numberchildren', 'over18', 'yearsincurrentrole'], errors='ignore')

    # Map numeric gender values to labels
    if 'gender' in df.columns:
        gender_map = {0: 'Female', 1: 'Male'}
        df['gender'] = df['gender'].map(gender_map).fillna(df['gender'])

    # Convert negative distances to absolute values
    if 'distancefromhome' in df.columns:
        df['distancefromhome'] = df['distancefromhome'].abs()

    # Fix typos in marital status
    if 'maritalstatus' in df.columns:
        df['maritalstatus'] = df['maritalstatus'].replace({'Marreid': 'Married'})

    # Convert selected columns to numeric if needed
    for col in ['dailyrate', 'monthlyrate', 'monthlyincome', 'hourlyrate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalize text values in key categorical columns
    cat_cols = ['jobrole', 'department', 'educationfield', 'overtime']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Convert binary columns to lowercase strings
    for col in ['attrition', 'remotework']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    return df

#%%
file = 'spaces_hr_raw_data.csv'
df = extract(file)

shape(df)
nulls(df,1,1)

df = df.drop(columns=['numberchildren', 'over18', 'yearsincurrentrole'])
df.shape
# %%

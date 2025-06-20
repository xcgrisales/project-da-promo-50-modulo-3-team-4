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
    df = df.drop(columns=['numberchildren', 'over18','yearsincurrentrole', 'Unnamed: 0',
                          'employeecount'], inplace=True, errors='ignore')

    # Words to numbers
    words_to_numbers = {'eighteen': 18,'nineteen': 19,'twenty': 20,'twenty-one': 21,'twenty-two': 22,
                        'twenty-three': 23, 'twenty-four': 24,'twenty-five': 25,'twenty-six': 26,
                        'twenty-seven': 27,'twenty-eight': 28,'twenty-nine': 29,'thirty': 30,
                        'thirty-one': 31,'thirty-two': 32,'thirty-three': 33,'thirty-four': 34,
                        'thirty-five': 35,'thirty-six': 36,'thirty-seven': 37,'forty-seven': 47,
                        'fifty-two': 52,'fifty-five': 55,'fifty-eight': 58}
    df['age'] = df['age'].replace(words_to_numbers)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Attrition
    df['attrition'] = df['attrition'].str.lower()

    #Business travel
    df['businesstravel'] = df['businesstravel'].str.replace(' ', '_', regex=False)
    df['businesstravel'] = df['businesstravel'].str.replace('-', '_', regex=False).str.lower()
    
    # Daily rate
    # reduce to two decimals
    df['dailyrate'] = df['dailyrate'].round(2)

    # Department
    df['department'] = df['department'].apply(
    lambda x: x.strip().replace(' ', '_').lower() if pd.notna(x) else x)

    # Distance
    # From negative to positives
    df['distancefromhome'] = df['distancefromhome'].abs()

    # Education Field
    df['educationfield'] = (
    df['educationfield']
    .astype(str)                      
    .str.strip('_')                   
    .str.replace('_', ' ', regex=False)  
    .str.strip()                      
    .str.lower()                     
    .str.replace(' ', '_', regex=False))

    # Dropping duplicates from 'employees' 
    df = df.drop_duplicates()
    # checking they are unique
    df['employeenumber'].is_unique

    # Environment satisfaction
    df['environmentsatisfaction'] = df['environmentsatisfaction'].apply(lambda x: np.nan if x>4 else x)

    satisfaction_map = {1: 'Low',
                        2: 'Medium',
                        3: 'High',
                        4: 'Very High'}
    df['environmentsatisfaction'] = df['environmentsatisfaction'].replace(satisfaction_map)

    # Gender
    gender_map = {0: 'Female',
                  1: 'Male'}
    
    df['gender'] = df['gender'].replace(gender_map)

    df['gender'] = df['gender'].astype(str).str.lower()

    # 

# %%

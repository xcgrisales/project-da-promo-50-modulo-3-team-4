#%%
#Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import itertools as it
import mysql.connector

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

from mysql.connector import errorcode
from sqlalchemy import create_engine

# configuration to visualize all the columns
pd.set_option('display.max_columns', None) 

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
'''
 _______                   __                           _   _             
|__   __|                 / _|                         | | (_)            
   | |_ __ __ _ _ __  ___| |_ ___  _ __ _ __ ___   __ _| |_ _  ___  _ __  
   | | '__/ _` | '_ \/ __|  _/ _ \| '__| '_ ` _ \ / _` | __| |/ _ \| '_ \ 
   | | | | (_| | | | \__ \ || (_) | |  | | | | | | (_| | |_| | (_) | | | |
   |_|_|  \__,_|_| |_|___/_| \___/|_|  |_| |_| |_|\__,_|\__|_|\___/|_| |_|
'''
# What is not working inside the main function
#Drop redundant or irrelevant columns
def drop_columns(df):
    df.drop(columns=['numberchildren', 'over18','yearsincurrentrole', 'Unnamed: 0',
                          'employeecount', 'sameasmonthlyincome','roledepartament'], inplace=True, errors='ignore')
    return df
#%%
# Transformation
def transform(df):
    # Age
    words_to_numbers = {'eighteen': '18',
                    'nineteen': '19',
                    'twenty': '20',
                    'twenty-one': '21',
                    'twenty-two': '22',
                    'twenty-three': '23',
                    'twenty-four': '24',
                    'twenty-five': '25',
                    'twenty-six': '26',
                    'twenty-seven': '27',
                    'twenty-eight': '28',
                    'twenty-nine': '29',
                    'thirty': '30',
                    'thirty-one': '31',
                    'thirty-two': '32',
                    'thirty-three': '33',
                    'thirty-four': '34',
                    'thirty-five': '35',
                    'thirty-six': '36',
                    'thirty-seven': '37',
                    'forty-seven': '47',
                    'fifty-two': '52',
                    'fifty-five': '55',
                    'fifty-eight': '58'}

    df['age'] = df['age'].replace(words_to_numbers)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Attrition
    df['attrition'] = df['attrition'].astype(str).str.lower()

    # Businesstravel
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

    # Hourly rate
    df['hourlyrate'] = df['hourlyrate'].round(2)

    # Job involvement
    jobinvolvement_map = {1: 'Low',
                          2: 'Medium',
                          3: 'High',
                          4: 'Very High'}
    
    df['jobinvolvement'] = df['jobinvolvement'].replace(jobinvolvement_map)

    df['jobinvolvement'] = (df['jobinvolvement'].astype(str).str.strip('_')
                            .str.replace('_', ' ', regex=False).str.strip()
                            .str.lower().str.replace(' ', '_', regex=False))

    # Job level
    joblevel_map = {1: 'Entry Level',
                    2: 'Intermediate',
                    3: 'Senior',
                    4: 'Manager',
                    5: 'Executive'}
    
    df['joblevel'] = df['joblevel'].replace(joblevel_map)

    df['joblevel'] = (df['joblevel']
                      .astype(str)
                      .str.strip('_')
                      .str.replace('_', ' ', regex=False)
                      .str.strip()
                      .str.lower()
                      .str.replace(' ', '_', regex=False))

    # Job role
    df['jobrole'] = df['jobrole'].str.strip().str.lower().str.title()

    df['jobrole'] = (df['jobrole']
                     .astype(str)
                     .str.strip('_')
                     .str.replace('_', ' ', regex=False)
                     .str.strip()
                     .str.lower()
                     .str.replace(' ', '_', regex=False))
    
    # Job satisfaction
    satisfaction_map = {1: 'Low',
                        2: 'Medium',
                        3: 'High',
                        4: 'Very High'}
    
    df['jobsatisfaction'] = df['jobsatisfaction'].replace(satisfaction_map)

    df['jobsatisfaction'] = (df['jobsatisfaction']
                             .astype(str)
                             .str.strip('_')
                             .str.replace('_', ' ', regex=False)
                             .str.strip()
                             .str.lower()
                             .str.replace(' ', '_', regex=False))
    
    # Marital status
    df['maritalstatus'] = df['maritalstatus'].str.strip().str.lower()
    df['maritalstatus'] = df['maritalstatus'].replace({'marreid': 'married'})

    # Monthly income
    df['monthlyincome'] = df['monthlyincome'].str.replace('$', '', regex=False)
    df['monthlyincome'] = df['monthlyincome'].str.replace(',', '.', regex=False)
    df['monthlyincome'] = pd.to_numeric(df['monthlyincome'], errors='coerce')

    # Monthly rate
    df['monthlyrate'] = df['monthlyrate'].astype(str)
    df['monthlyrate'] = df['monthlyrate'].str.replace('$', '', regex=False)
    df['monthlyrate'] = df['monthlyrate'].str.replace(',', '.', regex=False)
    df['monthlyrate'] = pd.to_numeric(df['monthlyrate'], errors='coerce')

    # Overtime
    df['overtime'] = df['overtime'].str.strip().str.lower()

    # Standard Hours
    df['standardhours'] = (df['standardhours']
                           .astype(str)
                           .str.strip('_')
                           .str.replace('_', ' ', regex=False)
                           .str.strip()
                           .str.lower()
                           .str.replace(' ', '_', regex=False))
    
    # Relationship Satisfaction
    rel_map = {1: 'Low',
               2: 'Medium',
               3: 'High',
               4: 'Very High'}
    
    df['relationshipsatisfaction'] = df['relationshipsatisfaction'].replace(rel_map)

    df['relationshipsatisfaction'] = (df['relationshipsatisfaction']
                                      .astype(str)
                                      .str.strip('_')
                                      .str.replace('_', ' ', regex=False)
                                      .str.strip()
                                      .str.lower()
                                      .str.replace(' ', '_', regex=False))
    
    # Total work in years
    df['totalworkingyears'].isna().sum()

    df['totalworkingyears'] = (df['totalworkingyears']
                               .astype(str)
                               .replace('<NA>', np.nan)
                               .str.replace(',', '.', regex=False)
                               .astype(float)
                               .astype('Int64'))
    
    # Work-life balance
    df['worklifebalance'].isna().sum()

    df['worklifebalance'] = df['worklifebalance'].str.replace(',', '.').astype(float)
    
    df['worklifebalance'] = df['worklifebalance'].replace({1.0: 'very_low',
                                                           2.0: 'low',
                                                           3.0: 'good',
                                                           4.0: 'excellent'})
    
    # Salary
    # make sure it is a text type
    df['salary'] = df['salary'].astype(str)
    # get rid of the symbol $ and replace comma for point
    df['salary'] = df['salary'].str.replace('$', '', regex=False)
    df['salary'] = df['salary'].str.replace(',', '.', regex=False)
    # Cnvert to decimal number (float)
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')

    # Remote work
    df['remotework'] = df['remotework'].astype(str).str.strip().str.lower()

    # replace variables for yes or no
    df['remotework'] = df['remotework'].replace({'1': 'yes',
                                                 'true': 'yes',
                                                 'yes': 'yes',
                                                 '0': 'no',
                                                 'false': 'no'})
    
    df = df.rename(columns={'businesstravel': 'business_travel', 
                        'dailyrate': 'daily_rate',
                        'distancefromhome': 'distance_from_home', 
                        'educationfield': 'education_field',
                        'employeecount': 'employee_count',
                        'employeenumber': 'employee_number',
                        'environmentsatisfaction': 'environment_satisfaction',
                        'hourlyrate': 'hourly_rate',
                        'jobinvolvement': 'job_involvement',
                        'joblevel': 'job_level',
                        'jobrole': 'job_role',
                        'jobsatisfaction': 'job_satisfaction',
                        'maritalstatus': 'marital_status',
                        'monthlyincome': 'monthly_income',
                        'monthlyrate': 'monthly_rate',
                        'numcompaniesworked': 'num_companies_worked',
                        'percentsalaryhike': 'percent_salary_hike',
                        'performancerating': 'performance_rating',
                        'relationshipsatisfaction': 'relationship_satisfaction',
                        'standardhours': 'standard_hours',
                        'stockoptionlevel': 'stock_option_level',
                        'totalworkingyears': 'total_working_years',
                        'trainingtimeslastyear': 'training_times_last_year',
                        'worklifebalance': 'work_life_balance',
                        'yearsatcompany': 'years_at_company',
                        'yearsincurrentrole': 'yearsincurrentrole',
                        'yearssincelastpromotion': 'years_since_last_promotion',
                        'yearswithcurrmanager': 'years_with_curr_manager',
                        'datebirth': 'date_birth', 
                        'roledepartament': 'role_departament',
                        'remotework': 'remote_work'})
    
    nulls(df,1,1)

    df.to_csv("df_transformado.csv", index=False)

'''_   _       _ _     
| \ | |     | | |    
|  \| |_   _| | |___ 
| . ` | | | | | / __|
| |\  | |_| | | \__ \
|_| \_|\__,_|_|_|___/

'''
# %%
def nulls_treatment(df):
    for col in df.columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            print(f"{col}: {nulls} nulls")

    nulls_percentaje = df.isnull().mean() * 100
    print(nulls_percentaje[nulls_percentaje > 0].sort_values(ascending=False))

    # overtime
    moda_overtime = df['overtime'].mode(dropna=True)[0]
    df['overtime'].fillna(moda_overtime, inplace=True)
    print(f"Remaining null observations in 'overtime': {df['overtime'].isnull().sum()}")

    # performance_rating
    moda_rating = df['performance_rating'].mode(dropna=True)[0]
    df['performance_rating'].fillna(moda_rating, inplace=True)
    print(f"Remaining null observations in 'performance_rating': {df['performance_rating'].isnull().sum()}")

    # standard_hours
    moda_hours = df['standard_hours'].mode(dropna=True)[0]
    df['standard_hours'].fillna(moda_hours, inplace=True)
    print(f"Remaining null observations in 'standard_hours': {df['standard_hours'].isnull().sum()}")

    # work_life_balance
    moda_wlb = df['work_life_balance'].mode(dropna=True)[0]
    df['work_life_balance'].fillna(moda_wlb, inplace=True)
    print(f"Remaining null observations in 'work_life_balance': {df['work_life_balance'].isnull().sum()}")

    # environment_satisfaction
    df['environment_satisfaction'].fillna('NA', inplace=True)
    print(f"Remaining null observations in 'environment_satisfaction': {df['environment_satisfaction'].isnull().sum()}")

    # Replacement with unknown category
    colums_unknown = ["department", "business_travel", "education_field", "marital_status"]
    for col in colums_unknown:
        df[col] = df[col].fillna("NA")
    print("After replacing using  'fillna' these are the remainig nulls")
    df[colums_unknown].isnull().sum()

    # Iterative and knn
    def iter_impute(df, columns):
            imputer_iter = IterativeImputer(max_iter=10, random_state=42)
            for col in columns:
                df[f'iter_{col}'] = imputer_iter.fit_transform(df[[col]])

    def knn_impute(df, columns):
        imputer_knn = KNNImputer(n_neighbors=5)
        for col in columns:
            df[f'knn_{col}'] = imputer_knn.fit_transform(df[[col]])

    cols_to_impute = ['hourly_rate', 'total_working_years', 'monthly_income', 'salary']
    iter_impute(df, cols_to_impute)
    knn_impute(df, cols_to_impute)

    # verifying
    for col in ['hourly_rate', 'total_working_years', 'monthly_income', 'salary']:
        print(f"iter_{col}: {df[f'iter_{col}'].isnull().sum()} nulos")
    for col in ['hourly_rate', 'total_working_years', 'monthly_income', 'salary']:
        print(f"knn_{col}: {df[f'knn_{col}'].isnull().sum()} nulos")

    #
    # Variables a imputar
    cols_to_impute = ['hourly_rate', 'total_working_years', 'monthly_income', 'salary']

    imputer_iter = IterativeImputer(max_iter=10, random_state=42)

    df_iter = pd.DataFrame(imputer_iter.fit_transform(df[cols_to_impute]), columns=cols_to_impute)
    df[cols_to_impute] = df_iter[cols_to_impute]
    print("Nulos restantes por variable:")
    print(df[cols_to_impute].isnull().sum())

    print(df.shape)

    for col in df.columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            print(f"{col}: {nulls} nulls")
        
    print('If nothing after shape - There are no nulls to fix! Finally!! :)')
    
    df.to_csv("df_final.csv", index=False)

#%%
''' _                     _ 
| |                   | |
| |     ___   __ _  __| |
| |    / _ \ / _` |/ _` |
| |___| (_) | (_| | (_| |
|______\___/ \__,_|\__,_|
'''
def table2(df):
    # Table 2 - Work Typologies
    # Create list of the typologies characteristics
    categories1 = df['business_travel'].unique().tolist()
    categories2 = df['standard_hours'].unique().tolist()
    categories3 = df['remote_work'].unique().tolist()

    # Create DataFrames (df) based on the litsts
    work_typologies1 = pd.DataFrame({'category_travel': categories1})
    work_typologies2 = pd.DataFrame({'category_std_hours': categories2})
    work_typologies3 = pd.DataFrame({'category_remote': categories3})

    # Create a mapping order for the characteristics
    order_mapping1 = {'non_travel': 1, 'travel_rarely': 2, 'travel_frequently': 3, 'NaN': 4}
    order_mapping2 = {'full_time': 1, 'part_time': 2, 'NaN': 3}
    order_mapping3 = {'yes': 1, 'no': 2, 'NaN': 3}

    # Sort using custom mapping
    work_typologies1 = work_typologies1.assign(sort_order = lambda work_typologies1: work_typologies1['category_travel'].map(order_mapping1)).sort_values('sort_order').drop('sort_order', axis=1)
    work_typologies2 = work_typologies2.assign(sort_order = lambda work_typologies2: work_typologies2['category_std_hours'].map(order_mapping2)).sort_values('sort_order').drop('sort_order', axis=1)
    work_typologies3 = work_typologies3.assign(sort_order = lambda work_typologies3: work_typologies3['category_remote'].map(order_mapping3)).sort_values('sort_order').drop('sort_order', axis=1)

    # Assign an _id
    work_typologies1.loc[work_typologies1['category_travel'] == 'non_travel', 'travel_id'] = 1
    work_typologies1.loc[work_typologies1['category_travel'] == 'travel_rarely', 'travel_id'] = 2
    work_typologies1.loc[work_typologies1['category_travel'] == 'travel_frequently', 'travel_id'] = 3

    work_typologies2.loc[work_typologies2['category_std_hours'] == 'full_time', 'std_hours_id'] = 1
    work_typologies2.loc[work_typologies2['category_std_hours'] == 'part_time', 'std_hours_id'] = 2


    work_typologies3.loc[work_typologies3['category_remote'] == 'yes', 'remote_id'] = 1
    work_typologies3.loc[work_typologies3['category_remote'] == 'no', 'remote_id'] = 2

    # Round _id, allowing for NaN values
    work_typologies1['travel_id'] = work_typologies1['travel_id'].round().astype('Int64') # Use 'Int64' to handle NaN values
    work_typologies2['std_hours_id'] = work_typologies2['std_hours_id'].round().astype('Int64') # Use 'Int64' to handle NaN values
    work_typologies3['remote_id'] = work_typologies3['remote_id'].round().astype('Int64') # Use 'Int64' to handle NaN values

    # Combine the characteristics
    cross_joined_df1 = work_typologies1.merge(work_typologies2, how = 'cross')
    cross_joined_df2 = cross_joined_df1.merge(work_typologies3, how = 'cross')
    cross_joined_df2 = cross_joined_df2.drop_duplicates()

    # Create typology_id based on characteristics' categories_id
    cross_joined_df2['typology_id'] = cross_joined_df2['travel_id']*100 + cross_joined_df2['std_hours_id']*10 + cross_joined_df2['remote_id'] 
    mask = pd.isna(cross_joined_df2['typology_id'])
    cross_joined_df2.loc[mask, 'typology_id'] = (cross_joined_df2.loc[mask, 'std_hours_id'] * 10 + cross_joined_df2.loc[mask, 'remote_id'])

    # Create typology based on characteristics' categories
    cross_joined_df2['typology'] = cross_joined_df2['category_travel'] + ', ' + cross_joined_df2['category_std_hours']+ ', ' + cross_joined_df2['category_remote'] +' remote' 
    mask = pd.isna(cross_joined_df2['typology'])
    cross_joined_df2.loc[mask, 'typology'] = (cross_joined_df2.loc[mask, 'category_std_hours'] + ', ' + cross_joined_df2.loc[mask, 'category_remote'] +' remote')

    print(cross_joined_df2[['travel_id','std_hours_id','remote_id','typology_id','typology']])

    table2 = cross_joined_df2.copy()

    table2.to_csv("typologies.csv", index=False)

    return table2

def table3(df):
    # Table 3 - Satisfaction_and_involvment
    # Create list of the satisfaction characteristics
    categories1 = df['environment_satisfaction'].unique().tolist()
    categories2 = df['job_satisfaction'].unique().tolist()
    categories3 = df['job_involvement'].unique().tolist()
    categories4 = df['relationship_satisfaction'].unique().tolist()
    categories5 = df['work_life_balance'].unique().tolist()

    # Create DataFrames (df) based on the litsts
    satisfaction_typologies1 = pd.DataFrame({'category_environment': categories1})
    satisfaction_typologies2 = pd.DataFrame({'category_job_satisfaction': categories2})
    satisfaction_typologies3 = pd.DataFrame({'category_job_involvment': categories3})
    satisfaction_typologies4 = pd.DataFrame({'category_relationship': categories4})
    satisfaction_typologies5 = pd.DataFrame({'category_balance': categories5})

    # Create a mapping order for the characteristics
    order_mapping1 = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4, 'NaN': 5}
    order_mapping5 = {'very_low': 1, 'low': 2, 'good': 3, 'excellent': 4, 'NaN': 5}

    # Sort using custom mapping
    satisfaction_typologies1 = satisfaction_typologies1.assign(sort_order = lambda satisfaction_typologies1: satisfaction_typologies1['category_environment'].map(order_mapping1)).sort_values('sort_order').drop('sort_order', axis=1)
    satisfaction_typologies2 = satisfaction_typologies2.assign(sort_order = lambda satisfaction_typologies2: satisfaction_typologies2['category_job_satisfaction'].map(order_mapping1)).sort_values('sort_order').drop('sort_order', axis=1)
    satisfaction_typologies3 = satisfaction_typologies3.assign(sort_order = lambda satisfaction_typologies3: satisfaction_typologies3['category_job_involvment'].map(order_mapping1)).sort_values('sort_order').drop('sort_order', axis=1)
    satisfaction_typologies4 = satisfaction_typologies4.assign(sort_order = lambda satisfaction_typologies4: satisfaction_typologies4['category_relationship'].map(order_mapping1)).sort_values('sort_order').drop('sort_order', axis=1)
    satisfaction_typologies5 = satisfaction_typologies5.assign(sort_order = lambda satisfaction_typologies5: satisfaction_typologies5['category_balance'].map(order_mapping5)).sort_values('sort_order').drop('sort_order', axis=1)

    # Assign an _id
    satisfaction_typologies1.loc[satisfaction_typologies1['category_environment'] == 'low', 'environment_id'] = 1
    satisfaction_typologies1.loc[satisfaction_typologies1['category_environment'] == 'medium', 'environment_id'] = 2
    satisfaction_typologies1.loc[satisfaction_typologies1['category_environment'] == 'high', 'environment_id'] = 3
    satisfaction_typologies1.loc[satisfaction_typologies1['category_environment'] == 'very_high', 'environment_id'] = 4
    satisfaction_typologies1.loc[satisfaction_typologies1['category_environment'] == 'NaN', 'environment_id'] = 5

    satisfaction_typologies2.loc[satisfaction_typologies2['category_job_satisfaction'] == 'low', 'job_satisfaction_id'] = 1
    satisfaction_typologies2.loc[satisfaction_typologies2['category_job_satisfaction'] == 'medium', 'job_satisfaction_id'] = 2
    satisfaction_typologies2.loc[satisfaction_typologies2['category_job_satisfaction'] == 'high', 'job_satisfaction_id'] = 3
    satisfaction_typologies2.loc[satisfaction_typologies2['category_job_satisfaction'] == 'very_high', 'job_satisfaction_id'] = 4
    satisfaction_typologies2.loc[satisfaction_typologies2['category_job_satisfaction'] == 'NaN', 'job_satisfaction_id'] = 5

    satisfaction_typologies3.loc[satisfaction_typologies3['category_job_involvment'] == 'low', 'job_involvment_id'] = 1
    satisfaction_typologies3.loc[satisfaction_typologies3['category_job_involvment'] == 'medium', 'job_involvment_id'] = 2
    satisfaction_typologies3.loc[satisfaction_typologies3['category_job_involvment'] == 'high', 'job_involvment_id'] = 3
    satisfaction_typologies3.loc[satisfaction_typologies3['category_job_involvment'] == 'very_high', 'job_involvment_id'] = 4
    satisfaction_typologies3.loc[satisfaction_typologies3['category_job_involvment'] == 'NaN', 'job_involvment_id'] = 5

    satisfaction_typologies4.loc[satisfaction_typologies4['category_relationship'] == 'low', 'relationship_id'] = 1
    satisfaction_typologies4.loc[satisfaction_typologies4['category_relationship'] == 'medium', 'relationship_id'] = 2
    satisfaction_typologies4.loc[satisfaction_typologies4['category_relationship'] == 'high', 'relationship_id'] = 3
    satisfaction_typologies4.loc[satisfaction_typologies4['category_relationship'] == 'very_high', 'relationship_id'] = 4
    satisfaction_typologies4.loc[satisfaction_typologies4['category_relationship'] == 'NaN', 'relationship_id'] = 4

    satisfaction_typologies5.loc[satisfaction_typologies5['category_balance'] == 'very_low', 'balance_id'] = 1
    satisfaction_typologies5.loc[satisfaction_typologies5['category_balance'] == 'low', 'balance_id'] = 2
    satisfaction_typologies5.loc[satisfaction_typologies5['category_balance'] == 'good', 'balance_id'] = 3
    satisfaction_typologies5.loc[satisfaction_typologies5['category_balance'] == 'excellent', 'balance_id'] = 4
    satisfaction_typologies5.loc[satisfaction_typologies5['category_balance'] == 'NaN', 'balance_id'] = 5

    # Round _id, allowing for NaN values
    satisfaction_typologies1['environment_id'] = satisfaction_typologies1['environment_id'].round().astype('Int64') # Use 'Int64' to handle NaN values
    satisfaction_typologies2['job_satisfaction_id'] = satisfaction_typologies2['job_satisfaction_id'].round().astype('Int64') # Use 'Int64' to handle NaN values
    satisfaction_typologies3['job_involvment_id'] = satisfaction_typologies3['job_involvment_id'].round().astype('Int64') # Use 'Int64' to handle NaN values
    satisfaction_typologies4['relationship_id'] = satisfaction_typologies4['relationship_id'].round().astype('Int64') # Use 'Int64' to handle NaN values
    satisfaction_typologies5['balance_id'] = satisfaction_typologies5['balance_id'].round().astype('Int64') # Use 'Int64' to handle NaN values

    # Combine the characteristics
    cross_joined_df1 = satisfaction_typologies1.merge(satisfaction_typologies2, how = 'cross')
    cross_joined_df2 = cross_joined_df1.merge(satisfaction_typologies3, how = 'cross')
    cross_joined_df3 = cross_joined_df2.merge(satisfaction_typologies4, how = 'cross')
    cross_joined_df4 = cross_joined_df3.merge(satisfaction_typologies5, how = 'cross')
    cross_joined_df4 = cross_joined_df4.drop_duplicates()

    # Create typology_id based on characteristics' categories_id
    cross_joined_df4['satisfaction_id'] = (cross_joined_df4['environment_id']*10000 + cross_joined_df4['job_satisfaction_id']*1000 + cross_joined_df4['job_involvment_id']*100 + 
                                        cross_joined_df4['relationship_id']*10 + cross_joined_df4['balance_id']*1)
    mask = pd.isna(cross_joined_df4['satisfaction_id'])
    cross_joined_df4.loc[mask, 'satisfaction_id'] = (cross_joined_df4.loc[mask, 'job_satisfaction_id'] * 1000 + cross_joined_df4.loc[mask, 'job_involvment_id']*100 + 
                                                cross_joined_df4.loc[mask, 'relationship_id'] * 10 + cross_joined_df4.loc[mask, 'balance_id'] * 1)

    # Create typology based on characteristics' categories
    cross_joined_df4['satisfaction'] = (cross_joined_df4['category_environment'] + ', ' + cross_joined_df4['category_job_satisfaction']+ ', ' + cross_joined_df4['category_job_involvment'] +
                                        ', ' + cross_joined_df4['category_relationship']+ ', ' + cross_joined_df4['category_balance'])
    mask = pd.isna(cross_joined_df4['satisfaction'])
    cross_joined_df4.loc[mask, 'satisfaction'] = ('environment_missing' + ' ,' + cross_joined_df4.loc[mask, 'category_job_satisfaction'] + ', ' + cross_joined_df4.loc[mask, 'category_job_involvment'] +
                                                ', ' + cross_joined_df4.loc[mask, 'category_relationship'] + ', ' + cross_joined_df4.loc[mask, 'category_balance'])

    cross_joined_df4[['environment_id','job_satisfaction_id','job_involvment_id','relationship_id','balance_id', 'satisfaction_id', 'satisfaction']]

    table3 = cross_joined_df4.copy()
    table3.to_csv("satisfaction_involvment.csv", index=False)
    return table3

def table4(df):
    # Table 4 - Department_role
    # Create list of the department_role categories
    categories1 = df['department'].unique().tolist()
    categories2 = df['job_level'].unique().tolist()
    categories3 = df['job_role'].unique().tolist()

    # Create DataFrames (df) based on the litsts
    department_role1 = pd.DataFrame({'category_department': categories1})
    department_role2 = pd.DataFrame({'category_level': categories2})
    department_role3 = pd.DataFrame({'category_role': categories3})

    # Create a mapping order for the characteristics
    order_mapping1 = {'research_&_development': 1, 'sales': 2, 'human_resources': 3, 'NaN': 4}
    order_mapping2 = {'entry_level': 1, 'manager': 2, 'executive': 3, 'intermediate': 4, 'senior': 5, 'NaN': 6}
    order_mapping3 = {'research_director': 1, 'manager': 2, 'sales_executive': 3, 'manufacturing_director': 4, 'research_scientist': 5,
                        'healthcare_representative': 6, 'laboratory_technician': 7, 'sales_representative': 8, 'human_resources': 9, 'NaN': 10}

    # Sort using custom mapping
    department_role1 = department_role1.assign(sort_order = lambda department_role1: department_role1['category_department'].map(order_mapping1)).sort_values('sort_order').drop('sort_order', axis=1)
    department_role2 = department_role2.assign(sort_order = lambda department_role2: department_role2['category_level'].map(order_mapping2)).sort_values('sort_order').drop('sort_order', axis=1)
    department_role3 = department_role3.assign(sort_order = lambda department_role3: department_role3['category_role'].map(order_mapping3)).sort_values('sort_order').drop('sort_order', axis=1)

    # Assign an _id
    department_role1.loc[department_role1['category_department'] == 'research_&_development', 'department_id'] = 1
    department_role1.loc[department_role1['category_department'] == 'sales', 'department_id'] = 2
    department_role1.loc[department_role1['category_department'] == 'human_resources', 'department_id'] = 3

    department_role2.loc[department_role2['category_level'] == 'entry_level', 'level_id'] = 1
    department_role2.loc[department_role2['category_level'] == 'manager', 'level_id'] = 2
    department_role2.loc[department_role2['category_level'] == 'executive', 'level_id'] = 3
    department_role2.loc[department_role2['category_level'] == 'intermediate', 'level_id'] = 4
    department_role2.loc[department_role2['category_level'] == 'senior', 'level_id'] = 5

    department_role3.loc[department_role3['category_role'] == 'research_director', 'role_id'] = 1
    department_role3.loc[department_role3['category_role'] == 'manager', 'role_id'] = 2
    department_role3.loc[department_role3['category_role'] == 'sales_executive', 'role_id'] = 3
    department_role3.loc[department_role3['category_role'] == 'manufacturing_director', 'role_id'] = 4
    department_role3.loc[department_role3['category_role'] == 'research_scientist', 'role_id'] = 5
    department_role3.loc[department_role3['category_role'] == 'healthcare_representative', 'role_id'] = 6
    department_role3.loc[department_role3['category_role'] == 'laboratory_technician', 'role_id'] = 7
    department_role3.loc[department_role3['category_role'] == 'sales_representative', 'role_id'] = 8
    department_role3.loc[department_role3['category_role'] == 'human_resources', 'role_id'] = 9

    # Round _id, allowing for NaN values
    department_role1['department_id'] = department_role1['department_id'].round().astype('Int64') # Use 'Int64' to handle NaN values
    department_role2['level_id'] = department_role2['level_id'].round().astype('Int64') # Use 'Int64' to handle NaN values
    department_role3['role_id'] = department_role3['role_id'].round().astype('Int64') # Use 'Int64' to handle NaN values

    # Combine the characteristics
    cross_joined_df1 = department_role1.merge(department_role2, how = 'cross')
    cross_joined_df2 = cross_joined_df1.merge(department_role3, how = 'cross')
    cross_joined_df2 = cross_joined_df2.drop_duplicates()

    # Create typology_id based on characteristics' categories_id
    cross_joined_df2['department_role_id'] = cross_joined_df2['department_id']*100 + cross_joined_df2['level_id']*10 + cross_joined_df2['role_id'] 
    mask = pd.isna(cross_joined_df2['department_role_id'])
    cross_joined_df2.loc[mask, 'department_role_id'] = (cross_joined_df2.loc[mask, 'level_id'] * 10 + cross_joined_df2.loc[mask, 'role_id'])

    # Create typology based on characteristics' categories
    cross_joined_df2['department_role'] = cross_joined_df2['category_department'] + ', ' + cross_joined_df2['category_level']+ ', ' + cross_joined_df2['category_role'] 
    mask = pd.isna(cross_joined_df2['department_role'])
    cross_joined_df2.loc[mask, 'department_role'] = (cross_joined_df2.loc[mask, 'category_level'] + ', ' + cross_joined_df2.loc[mask, 'category_role'])

    print(cross_joined_df2[['department_id','level_id','role_id','department_role_id','department_role']])

    table4 = cross_joined_df2.copy() 
    table4.to_csv("department_role.csv", index=False)
    return table4

def table1(df, table2, table3, table4):
    # Table 1
    df_employees = df.copy()
    df_employees.columns
    df_employees.rename(columns={"business_travel": "category_travel", "standard_hours": "category_std_hours", 'remote_work': 'category_remote',
                    'environment_satisfaction': 'category_environment', 'job_satisfaction': 'category_job_satisfaction', 'job_involvement': 'category_job_involvment',
                    'relationship_satisfaction': 'category_relationship', 'work_life_balance': 'category_balance',
                    'department': 'category_department', 'job_level': 'category_level', 'job_role': 'category_role'}, inplace=True)


    df_employees = df_employees.merge(table2, on=['category_travel', 'category_std_hours', 'category_remote'], how='left')
    df_employees = df_employees.merge(table3, on=['category_environment', 'category_job_satisfaction', 'category_job_involvment', 'category_relationship', 'category_balance'], how='left')
    df_employees = df_employees.merge(table4, on=['category_department', 'category_level', 'category_role'], how='left')


    df_employees = df_employees.loc[:, ~df_employees.columns.str.startswith('category_')]

    columns = ['travel_id', 'std_hours_id', 'remote_id','environment_id', 'job_satisfaction_id', 'job_involvment_id','relationship_id',
            'balance_id', 'department_id', 'level_id', 'role_id', 'typology','satisfaction','department_role']
    df_employees = df_employees.drop(columns=columns)

    table1 = df_employees.copy() 
    table1.to_csv("employees.csv", index=False)
    
    return table1

def trying_connection():
    user = 'root'
    password = 'AlumnaAdalab'
    host = '127.0.0.1'

    # Try the connection and check the error message and code in case it is not working
    try:
        cnx = mysql.connector.connect(user = user, password = password,
                                        host = host)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        cnx.close()

def drop_table():
    import mysql.connector

    # Connection parameters
    user = 'root'
    password = 'AlumnaAdalab'
    host = '127.0.0.1'

    # Connect to MySQL server
    cnx = mysql.connector.connect(user=user, password=password, host=host)
    mycursor = cnx.cursor()

    # Try to drop the database
    try:
        mycursor.execute("DROP DATABASE project_talent")
        print("Database 'project_talent' dropped successfully.")
    except mysql.connector.Error as err:
        print("Failed to drop database:")
        print("Error Code:", err.errno)
        print("SQLSTATE", err.sqlstate)
        print("Message", err.msg)

    # Close connection
    mycursor.close()
    cnx.close()

def create_table():
    user = 'root'
    password = 'AlumnaAdalab'
    host = '127.0.0.1'

    # conexión (1)
    cnx = mysql.connector.connect(user = user, password = password,
                                host = host)

    # cursor (2)
    mycursor = cnx.cursor()

    try:
        mycursor.execute("CREATE DATABASE project_talent")
        print(mycursor)
    except mysql.connector.Error as err:
        print(err)
        print("Error Code:", err.errno)
        print("SQLSTATE", err.sqlstate)
        print("Message", err.msg)

def loading_employees():
    # Adding the data from employee.csv - data on employees
    engine_route = 'mysql+mysqlconnector://root:AlumnaAdalab@127.0.0.1/project_talent'
    location = 'employees.csv'

    # Create the connection engine
    engine = create_engine(engine_route)
    # Uppload the CSV file in a DataFrame 
    df = pd.read_csv(location)
    # Verify that the first registries are in order
    print(df.head())

    # send the data to 
    df.to_sql("employees", engine, if_exists="append", index=False)

def loading_typologies():
    # Adding the data from typologies.csv - data on employment characteristics
    engine_route = 'mysql+mysqlconnector://root:AlumnaAdalab@127.0.0.1/project_talent'
    location = 'typologies.csv'

    # Create the connection engine
    engine = create_engine(engine_route)
    # Uppload the CSV file in a DataFrame 
    df = pd.read_csv(location)
    # Verify that the first registries are in order
    print(df.head())

    df.to_sql("typologies", engine, if_exists="append", index=False)

def loading_satisfaction():
    # Adding the data from satisfaction_involment.csv - data on satisfaction metrics
    engine_route = 'mysql+mysqlconnector://root:AlumnaAdalab@127.0.0.1/project_talent'
    location = 'satisfaction_involvment.csv'

    # Create the connection engine
    engine = create_engine(engine_route)
    # Uppload the CSV file in a DataFrame 
    df = pd.read_csv(location)
    # Verify that the first registries are in order
    print(df.head())

    df.to_sql("satisfaction_involvment", engine, if_exists="append", index=False)

def loading_department_role():
    # Adding the data from department_role.csv - data on department and role classified
    engine_route = 'mysql+mysqlconnector://root:AlumnaAdalab@127.0.0.1/project_talent'
    location = 'department_role.csv'

    # Create the connection engine
    engine = create_engine(engine_route)
    # Uppload the CSV file in a DataFrame 
    df = pd.read_csv(location)
    # Verify that the first registries are in order
    print(df.head())

    df.to_sql("department_role", engine, if_exists="append", index=False)


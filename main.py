#%%
import etl 

#%%
file = 'spaces_hr_raw_data.csv'
df = etl.extract(file)

etl.shape(df)
etl.nulls(df,1,1)

#%%
df = df.drop(columns=['numberchildren', 'over18', 'yearsincurrentrole'])
df.shape

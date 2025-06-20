#%%
import etl 

#%%
file = 'spaces_hr_raw_data.csv'
df = etl.extract(file)

etl.shape(df)
etl.nulls(df,1,1)
df = etl.drop_columns(df)

etl.transform(df)

# %%
file = 'df_transformado.csv'
df = etl.extract(file)

etl.nulls_treatment(df)
# %%
file = 'df_final.csv'

df = etl.extract(file)

# %%
table2 = etl.table2(df)
table3 = etl.table3(df)
table4 = etl.table4(df)
table1 = etl.table1(df,table2,table3,table4)

etl.trying_connection()
etl.drop_table()
etl.create_table()

etl.loading_employees()
etl.loading_typologies()
etl.loading_satisfaction()
etl.loading_department_role()


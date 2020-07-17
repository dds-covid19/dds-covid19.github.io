import pandas as pd
import numpy as np
from datetime import datetime as dt

# process and save the covid data
df = (pd.read_csv("data/time_series_covid19_deaths_US.csv"))
df1 = df.groupby(['Province_State']).sum()
df1 = df1.loc[:,'1/22/20':]
df1 = df1.drop(index=['Guam','Virgin Islands','Puerto Rico','Grand Princess','American Samoa','Northern Mariana Islands','Diamond Princess'])
new_death_cases = df1 - df1.shift(1,axis=1)
new_death_cases[new_death_cases<0]=0
df = pd.read_csv("data/time_series_covid19_confirmed_US.csv")
df2 = df.groupby(['Province_State']).sum()
df2 = df2.loc[:,'1/22/20':]
df2 = df2.drop(index=['Guam','Virgin Islands','Puerto Rico','Grand Princess','American Samoa','Northern Mariana Islands','Diamond Princess'])
new_daily_cases = df2 - df2.shift(1,axis=1)
new_daily_cases[new_daily_cases<0]=0
new_death_cases.to_csv ('data/new_death_cases.csv', index = True, header=True)
new_daily_cases.to_csv ('data/new_daily_cases.csv', index = True, header=True)






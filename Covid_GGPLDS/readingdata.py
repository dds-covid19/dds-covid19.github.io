#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:18:24 2020

@author: rahikalantari
"""

#import pandas as pd
#from datetime import datetime as dt
#df = (pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")
#    .assign(date=lambda df: [dt.strptime(x, "%Y-%m-%d") for x in df.date])
#    .groupby(["date", "state"])
#    .agg({"cases": sum, "deaths": sum})
#    .reset_index())
#df.tail()

import pandas as pd
import numpy as np
from datetime import datetime as dt
df = (pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"))
df1 = df.groupby(['Province_State']).sum()
df1 = df1.loc[:,'1/22/20':]
df1 = df1.drop(index=['Guam','Virgin Islands','Puerto Rico','Grand Princess','American Samoa','Northern Mariana Islands','Diamond Princess'])
new_death_cases = df1 - df1.shift(1,axis=1)
new_death_cases[new_death_cases<0]=0
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
df2 = df.groupby(['Province_State']).sum()
df2 = df2.loc[:,'1/22/20':]
df2 = df2.drop(index=['Guam','Virgin Islands','Puerto Rico','Grand Princess','American Samoa','Northern Mariana Islands','Diamond Princess'])
new_daily_cases = df2 - df2.shift(1,axis=1)
new_daily_cases[new_daily_cases<0]=0
new_death_cases.to_csv ('data/new_death_cases.csv', index = True, header=True)
new_daily_cases.to_csv ('data/new_daily_cases.csv', index = True, header=True)

death_mean = pd.read_csv('results/Historic_prediction/death_mean_.csv')
death_mean = death_mean.rename(columns=death_mean.loc[0,:])
death_mean=(death_mean.drop(index=0))#.drop('01/22/0020',axis=1)
death_lower = pd.read_csv('results/Historic_prediction/death_lowerBound_.csv')
death_lower = death_lower.rename(columns=death_lower.loc[0,:])
death_lower = (death_lower.drop(index=0))#.drop('01/22/0020',axis=1)
death_higher = pd.read_csv('results/Historic_prediction/death_upperBound_.csv')
death_higher = death_higher.rename(columns=death_higher.loc[0,:])
death_higher = (death_higher.drop(index=0))#.drop('01/22/0020',axis=1)
for i in range(1,53):
    death_mean.loc[i,'3/15/20':] =  pd.to_numeric(death_mean.loc[i,'3/15/20':],errors='coerce')
    death_lower.loc[i,'3/15/20':] =  pd.to_numeric(death_lower.loc[i,'3/15/20':],errors='coerce')
    death_higher.loc[i,'3/15/20':] =  pd.to_numeric(death_higher.loc[i,'3/15/20':],errors='coerce')
    
death_mean.to_csv ('results/Historic_prediction/death_mean_.csv', index = False, header=True)
death_lower.to_csv ('results/Historic_prediction/death_lowerBound_.csv', index = False, header=True)
death_higher.to_csv ('results/Historic_prediction/death_upperBound_.csv', index = False, header=True)

death_mean = pd.read_csv('results/Future_prediction/death_mean_.csv')
death_mean = death_mean.rename(columns=death_mean.loc[0,:])
death_mean=(death_mean.drop(index=0)).drop('01/22/0020',axis=1)
death_lower = pd.read_csv('results/Future_prediction/death_lowerBound_.csv')
death_lower = death_lower.rename(columns=death_lower.loc[0,:])
death_lower = (death_lower.drop(index=0)).drop('01/22/0020',axis=1)
death_higher = pd.read_csv('results/Future_prediction/death_upperBound_.csv')
death_higher = death_higher.rename(columns=death_higher.loc[0,:])
death_higher = (death_higher.drop(index=0)).drop('01/22/0020',axis=1)
for i in range(1,53):
    death_mean.loc[i,'1/23/20':] =  pd.to_numeric(death_mean.loc[i,'1/23/20':],errors='coerce')
    death_lower.loc[i,'1/23/20':] =  pd.to_numeric(death_lower.loc[i,'1/23/20':],errors='coerce')
    death_higher.loc[i,'1/23/20':] =  pd.to_numeric(death_higher.loc[i,'1/23/20':],errors='coerce')
    
    
death_mean.to_csv ('results/Future_prediction/death_mean_.csv', index = True, header=True)
death_lower.to_csv ('results/Future_prediction/death_lowerBound_.csv', index = True, header=True)
death_higher.to_csv ('results/Future_prediction/death_upperBound_.csv', index = True, header=True)

death_mean_col = pd.melt(death_mean, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths')
death_lower_col = pd.melt(death_lower, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_lower')
death_higher_col = pd.melt(death_higher, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_higher')
results_death= pd.merge(death_mean_col, death_lower_col, how='outer', on=['Province_State', 'date'])
results_death= pd.merge(results_death, death_higher_col, how='outer', on=['Province_State', 'date'])
results_death.to_csv ('results/Future_prediction/furture_death_.csv', index = True, header=True)

death_mean = pd.read_csv('results/Historic_prediction/death_mean_.csv')
death_lower = pd.read_csv('results/Historic_prediction/death_lowerBound_.csv')
death_higher = pd.read_csv('results/Historic_prediction/death_upperBound_.csv')
death_mean_col = pd.melt(death_mean, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths')
death_lower_col = pd.melt(death_lower, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_lower')
death_higher_col = pd.melt(death_higher, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_higher')
results_death= pd.merge(death_mean_col, death_lower_col, how='outer', on=['Province_State', 'date'])
results_death= pd.merge(results_death, death_higher_col, how='outer', on=['Province_State', 'date'])
results_death.to_csv ('results/Historic_prediction/historic_death_.csv', index = True, header=True)


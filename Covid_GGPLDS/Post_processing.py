#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 05:29:48 2020

@author: rahikalantari
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt


task = 'future' #'historic''future'

event = 'death'#'cases''death'


if task == 'historic':
    foldername = 'Historic'
  
else:
    foldername = 'Future'
    TP = -23




if event == 'death':
    death_mean = pd.read_csv('results/'+foldername+'_prediction/death_mean_.csv')
    death_mean = death_mean.rename(columns=death_mean.loc[0,:])

    death_mean = death_mean.iloc[:,:]

    death_mean = (death_mean.drop(index=0))#.drop('01/22/0020',axis=1)
    death_lower = pd.read_csv('results/'+foldername+'_prediction/death_lowerBound_.csv')
    death_lower = death_lower.rename(columns=death_lower.loc[0,:])
    death_lower = (death_lower.drop(index=0))#.drop('01/22/0020',axis=1)
    death_lower = death_lower.iloc[:,:-23]
    death_higher = pd.read_csv('results/'+foldername+'_prediction/death_upperBound_.csv')
    death_higher = death_higher.rename(columns=death_higher.loc[0,:])

    death_higher = (death_higher.drop(index=0))
    death_higher= death_higher.iloc[:,:]


#    death_mean=(death_mean.drop(index=0))#.drop('01/22/0020',axis=1)
#    death_lower = pd.read_csv('results/'+foldername+'_prediction/death_lowerBound_.csv')
#    death_lower = death_lower.rename(columns=death_lower.loc[0,:])
#    death_lower = (death_lower.drop(index=0))#.drop('01/22/0020',axis=1)
#    death_higher = pd.read_csv('results/'+foldername+'_prediction/death_upperBound_.csv')
#    death_higher = death_higher.rename(columns=death_higher.loc[0,:])
#    death_higher = (death_higher.drop(index=0))#.drop('01/22/0020',axis=1)
    realdata_death = realdata = pd.read_csv('data/new_death_cases.csv')
    
    
    for i in range(1,53):
        death_mean.loc[i,'3/15/20':] =  pd.to_numeric(death_mean.loc[i,'3/15/20':],errors='coerce')
        death_lower.loc[i,'3/15/20':] =  pd.to_numeric(death_lower.loc[i,'3/15/20':],errors='coerce')
        death_higher.loc[i,'3/15/20':] =  pd.to_numeric(death_higher.loc[i,'3/15/20':],errors='coerce')

    death_mean.to_csv ('results/'+foldername+'_prediction/death_mean_2020_08_10.csv', index = False, header=True)
    death_lower.to_csv ('results/'+foldername+'_prediction/death_lowerBound_2020_08_10.csv', index = False, header=True)
    death_higher.to_csv ('results/'+foldername+'_prediction/death_upperBound_2020_08_10.csv', index = False, header=True)

    realdata_death.loc[51] = realdata_death.sum(axis=0)
    realdata_death["Province_State"].loc[51] = "US"
    
    real_death_col = pd.melt(realdata_death, id_vars=['Province_State'], var_name='date', value_name='real_number_of_deaths')
    
    
   # results_death.to_csv ('results/Future_prediction/furture_death_.csv', index = True, header=True)
    

    death_mean = pd.read_csv('results/'+foldername+'_prediction/death_mean_2020_08_10.csv')


    death_mean.loc[51] = death_mean.loc[0:50].sum(axis=0)
    death_mean["Province_State"].loc[51] = "US"


    death_lower = pd.read_csv('results/'+foldername+'_prediction/death_lowerBound_2020_08_10.csv')
    death_higher = pd.read_csv('results/'+foldername+'_prediction/death_upperBound_2020_08_10.csv')

    death_mean_col = pd.melt(death_mean, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths')
    death_lower_col = pd.melt(death_lower, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_lower')
    death_higher_col = pd.melt(death_higher, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_higher')
    results_death= pd.merge(real_death_col, death_mean_col, how='outer', on=['Province_State', 'date'])
    results_death= pd.merge(results_death, death_lower_col, how='outer', on=['Province_State', 'date'])
    results_death= pd.merge(results_death, death_higher_col, how='outer', on=['Province_State', 'date'])
    #results_death= pd.merge(results_death, real_death_col, how='outer', on=['Province_State', 'date'])

    results_death.to_csv ('results/'+foldername+'_prediction/'+foldername+'__death_2020_08_10.csv', index = True, header=True)

    


    
    
else:
    cases_mean = pd.read_csv('results/'+foldername+'_prediction/daily_cases_mean_.csv')
    cases_mean = cases_mean.rename(columns=cases_mean.iloc[0,:])

    cases_mean = cases_mean.iloc[:,:]
    cases_mean=(cases_mean.drop(index=0))#.drop('01/22/0020',axis=1)
    cases_lower = pd.read_csv('results/'+foldername+'_prediction/daily_cases_lowerBound_.csv')
    cases_lower = cases_lower.rename(columns=cases_lower.iloc[0,:])
    cases_lower = cases_lower.iloc[:,:]
    cases_lower = (cases_lower.drop(index=0))#.drop('01/22/0020',axis=1)
    cases_higher = pd.read_csv('results/'+foldername+'_prediction/daily_cases_upperBound_.csv')
    cases_higher = cases_higher.rename(columns=cases_higher.iloc[0,:])
    cases_higher = cases_higher.iloc[:,:]

    cases_higher = (cases_higher.drop(index=0))#.drop('01/22/0020',axis=1)
    realdata_cases = realdata = pd.read_csv('data/new_daily_cases.csv')
    
    
    for i in range(1,53):
        cases_mean.loc[i,'3/15/20':] =  pd.to_numeric(cases_mean.loc[i,'3/15/20':],errors='coerce')
        cases_lower.loc[i,'3/15/20':] =  pd.to_numeric(cases_lower.loc[i,'3/15/20':],errors='coerce')
        cases_higher.loc[i,'3/15/20':] =  pd.to_numeric(cases_higher.loc[i,'3/15/20':],errors='coerce')
    
    cases_mean.to_csv ('results/'+foldername+'_prediction/daily_cases_mean_2020_08_10.csv', index = False, header=True)
    cases_lower.to_csv ('results/'+foldername+'_prediction/daily_cases_lowerBound_2020_08_10.csv', index = False, header=True)
    cases_higher.to_csv ('results/'+foldername+'_prediction/daily_cases_upperBound_2020_08_10.csv', index = False, header=True)

    realdata_cases.loc[51] = realdata_cases.sum(axis=0)
    realdata_cases["Province_State"].loc[51] = "US"
    
    real_cases_col = pd.melt(realdata_cases, id_vars=['Province_State'], var_name='date', value_name='real_number_of_daily_cases')
#    cases_mean_col = pd.melt(cases_mean, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths')
#    cases_lower_col = pd.melt(cases_lower, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_lower')
#    cases_higher_col = pd.melt(cases_higher, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_higher')
#    results_cases= pd.merge(real_cases_col, cases_mean_col, how='outer', on=['Province_State', 'date'])
#    results_cases= pd.merge(results_cases, cases_lower_col, how='outer', on=['Province_State', 'date'])
#    results_cases= pd.merge(results_cases, cases_higher_col, how='outer', on=['Province_State', 'date'])
#    #results_death= pd.merge(results_death, real_death_col, how='outer', on=['Province_State', 'date'])
#    results_cases.to_csv ('results/'+foldername+'_prediction/furture_death_.csv', index = True, header=True)
    

    cases_mean = pd.read_csv('results/'+foldername+'_prediction/daily_cases_mean_2020_08_10.csv')


    cases_mean.loc[51] = cases_mean.loc[0:50].sum(axis=0)
    cases_mean["Province_State"].loc[51] = "US"


    cases_lower = pd.read_csv('results/'+foldername+'_prediction/daily_cases_lowerBound_2020_08_10.csv')
    cases_higher = pd.read_csv('results/'+foldername+'_prediction/daily_cases_upperBound_2020_08_10.csv')

    cases_mean_col = pd.melt(cases_mean, id_vars=['Province_State'], var_name='date', value_name='number_of_daily_cases')
    cases_lower_col = pd.melt(cases_lower, id_vars=['Province_State'], var_name='date', value_name='number_of_daily_cases_lower')
    cases_higher_col = pd.melt(cases_higher, id_vars=['Province_State'], var_name='date', value_name='number_of_daily_cases_higher')
    results_cases= pd.merge(real_cases_col, cases_mean_col, how='outer', on=['Province_State', 'date'])
    results_cases= pd.merge(results_cases, cases_lower_col, how='outer', on=['Province_State', 'date'])
    results_cases= pd.merge(results_cases, cases_higher_col, how='outer', on=['Province_State', 'date'])
#    #results_death= pd.merge(results_death, real_death_col, how='outer', on=['Province_State', 'date'])
#    results_death= pd.merge(death_mean_col, death_lower_col, how='outer', on=['Province_State', 'date'])
#    results_death= pd.merge(results_death, death_higher_col, how='outer', on=['Province_State', 'date'])

    results_cases.to_csv ('results/'+foldername+'_prediction/'+foldername+'__daily_cases_2020_08_10.csv', index = True, header=True)

    

#    death_mean = pd.read_csv('results/Future_prediction/death_mean_.csv')
#    death_mean = death_mean.rename(columns=death_mean.loc[0,:])
#    death_mean=(death_mean.drop(index=0))#.drop('01/22/0020',axis=1)
#    death_lower = pd.read_csv('results/Future_prediction/death_lowerBound_.csv')
#    death_lower = death_lower.rename(columns=death_lower.loc[0,:])
#    death_lower = (death_lower.drop(index=0))#.drop('01/22/0020',axis=1)
#    death_higher = pd.read_csv('results/Future_prediction/death_upperBound_.csv')
#    death_higher = death_higher.rename(columns=death_higher.loc[0,:])
#    death_higher = (death_higher.drop(index=0))#.drop('01/22/0020',axis=1)
#
#
#for i in range(1,53):
#    death_mean.loc[i,'3/15/20':] =  pd.to_numeric(death_mean.loc[i,'3/15/20':],errors='coerce')
#    death_lower.loc[i,'3/15/20':] =  pd.to_numeric(death_lower.loc[i,'3/15/20':],errors='coerce')
#    death_higher.loc[i,'3/15/20':] =  pd.to_numeric(death_higher.loc[i,'3/15/20':],errors='coerce')
#    
#    
#death_mean.to_csv ('results/Future_prediction/death_mean_.csv', index = False, header=True)
#death_lower.to_csv ('results/Future_prediction/death_lowerBound_.csv', index = False, header=True)
#death_higher.to_csv ('results/Future_prediction/death_upperBound_.csv', index = False, header=True)


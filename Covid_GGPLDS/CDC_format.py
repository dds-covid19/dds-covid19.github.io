#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:18:30 2020

@author: rahikalantari
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 05:29:48 2020

@author: rahikalantari
"""
import datetime
import pandas as pd
import numpy as np
from datetime import datetime as dt

task = 'future' #'historic''future'
event = 'death'#'cases''death'
if task == 'historic':
    foldername = 'Historic'
else:
    foldername = 'Future'
    
us_state_code = {
'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05', 'California': '06', 'Colorado': '08',
'Connecticut': '09', 'Delaware': '10', 'Florida': '12', 'Georgia': '13', 'Hawaii': '15', 'Idaho': '16',
'Illinois': '17', 'Indiana': '18', 'Iowa': '19', 'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22',
'Maine': '23', 'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27', 'Mississippi': '28',
'Missouri': '29', 'Montana': '30', 'Nebraska': '31', 'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34',
'New Mexico': '35', 'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39', 'Oklahoma': '40',
'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44', 'South Carolina': '45', 'South Dakota': '46',
'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50', 'Virginia': '51', 'Washington': '53',
'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56', 'District of Columbia':'11'}
    
task = 'future' #'historic''future'
event = 'death'#'cases''death'
if task == 'historic':
    foldername = 'Historic'
else:
    foldername = 'Future'

if event == 'death':
    death_mean = pd.read_csv('results/'+foldername+'_prediction/death_cum.csv')
    death_weekly  =  pd.read_csv('results/'+foldername+'_prediction/death_weekly.csv')
    daily_cases_weekly  =  pd.read_csv('results/'+foldername+'_prediction/daily_cases_weekly.csv')
    
    death_mean = death_mean.rename(columns=death_mean.loc[0,:])
    death_mean=(death_mean.drop(index=0))#.drop('01/22/0020',axis=1)
    death_weekly = death_weekly.rename(columns=death_weekly.loc[0,:])
    death_weekly=(death_weekly.drop(index=0))#.drop('01/22/0020',axis=1)
    daily_cases_weekly = daily_cases_weekly.rename(columns= daily_cases_weekly.loc[0,:])
    daily_cases_weekly = (daily_cases_weekly.drop(index=0))#.drop('01/22/0020',axis=1)
    #daily_cases_weekly =  daily_cases_weekly_col['type'].loc[daily_cases_weekly_col['type'] =='NA' | daily_cases_weekly_col['type']=='0.025' | daily_cases_weekly_col['type']=='0.1'| daily_cases_weekly_col['type']== '0.25'| daily_cases_weekly_col['type']=='0.500'| daily_cases_weekly_col['type']== '0.750'| daily_cases_weekly_col['type']=='0.900'| daily_cases_weekly_col['type']=='0.975']
    daily_cases_weekly_col1 = []
    

    #aily_cases_weekly  =  pd.read_csv('results/'+foldername+'_prediction/daily_cases_weekly.csv')
#    death_lower = pd.read_csv('results/'+foldername+'_prediction/death_lowerBound_cum.csv')
#    death_lower = death_lower.rename(columns=death_lower.loc[0,:])
#    death_lower = (death_lower.drop(index=0))#.drop('01/22/0020',axis=1)
#    death_higher = pd.read_csv('results/'+foldername+'_prediction/death_upperBound_cum.csv')
#    death_higher = death_higher.rename(columns=death_higher.loc[0,:])
#    death_higher = (death_higher.drop(index=0))#.drop('01/22/0020',axis=1)

    realdata_death = realdata = pd.read_csv('data/new_death_cases.csv')
    
    
    for i in range(1,53):
        death_mean.loc[i,'3/15/20':] =  pd.to_numeric(death_mean.loc[i,'3/15/20':],errors='coerce')
        death_weekly.loc[i,'3/15/20':] =  pd.to_numeric(death_weekly.loc[i,'3/15/20':],errors='coerce')
        daily_cases_weekly.loc[i,'3/15/20':] =  pd.to_numeric(daily_cases_weekly.loc[i,'3/15/20':],errors='coerce')
        
    
    death_mean.to_csv ('results/'+foldername+'_prediction/death_cum1.csv', index = False, header=True)
    death_weekly.to_csv ('results/'+foldername+'_prediction/death_weekly1.csv', index = False, header=True)
    daily_cases_weekly.to_csv ('results/'+foldername+'_prediction/daily_cases_weekly1.csv', index = False, header=True)
    
    realdata_death.loc[51] = realdata_death.sum(axis=0)
    realdata_death["Province_State"].loc[51] = "US"
    
    real_death_col = pd.melt(realdata_death, id_vars=['Province_State'], var_name='date', value_name='real_number_of_deaths')
    
    
   # results_death.to_csv ('results/Future_prediction/furture_death_.csv', index = True, header=True)
    
    death_mean = pd.read_csv('results/'+foldername+'_prediction/death_cum1.csv')
    
    death_mean_col = pd.melt(death_mean, id_vars=['Province_State','type'], var_name='date', value_name='number_of_deaths')
    
    results_death= death_mean_col#pd.merge(real_death_col, death_mean_col, how='outer', on=['Province_State', 'date'])
    
    death_weekly = pd.read_csv('results/'+foldername+'_prediction/death_weekly1.csv')
    
    death_weekly_col = pd.melt(death_weekly, id_vars=['Province_State','type'], var_name='date', value_name='number_of_deaths')
    
    results_death_weekly= death_weekly_col#pd.merge(real_death_col, death_mean_col, how='outer', on=['Province_State', 'date'])

   
    daily_cases_weekly = pd.read_csv('results/'+foldername+'_prediction/daily_cases_weekly1.csv')
    
    daily_cases_weekly_col = pd.melt(daily_cases_weekly, id_vars=['Province_State','type'], var_name='date', value_name='number_of_deaths')
    daily_cases_weekly_col1= pd.DataFrame(columns=daily_cases_weekly_col.columns)
    for qq in ['point', '0.025', '0.1', '0.25','0.5','0.75','0.9','0.975']:
        temp = daily_cases_weekly_col.loc[daily_cases_weekly_col['type'] ==qq]
        daily_cases_weekly_col1 = daily_cases_weekly_col1.append(temp)
    results_daily_cases_weekly= daily_cases_weekly_col1#pd.merge(real_death_col, death_mean_col, how='outer', on=['Province_State', 'date'])

    #results_death= pd.merge(results_death, real_death_col, how='outer', on=['Province_State', 'date'])
    #results_death.to_csv ('results/'+foldername+'_prediction/'+foldername+'__death_cum.csv', index = True, header=True)
    
    #daily_cases_weekly = pd.read_csv('results/'+foldername+'_prediction/daily_cases_weekly1.csv')
    

    #daily_cases_weekly_col = pd.melt(daily_cases_weekly, id_vars=['Province_State','type'], var_name='date', value_name='number_of_deaths')
    
    #results_daily_cases_weekly= daily_cases_weekly_col#pd.merge(real_death_col, death_mean_col, how='outer', on=['Province_State', 'date'])
    #results_death= pd.merge(results_death, real_death_col, how='outer', on=['Province_State', 'date'])
    #results_death.to_csv ('results/'+foldername+'_prediction/'+foldername+'__death_cum.csv', index = True, header=True)




if event == 'death':
    
    if task == 'future':
        
        #death_mean.style.format({"date": lambda t: t.strftime("%Y-%m-%d")}) 
        results_death['date'] = pd.to_datetime(results_death.date).dt.strftime('%Y-%m-%d')
        results_death_weekly['date'] = pd.to_datetime(results_death_weekly.date).dt.strftime('%Y-%m-%d')
        results_daily_cases_weekly['date'] = pd.to_datetime(results_death_weekly.date).dt.strftime('%Y-%m-%d')

        today= dt.today() 
        #today= dt.today() - datetime.timedelta(days=2)
        today_date = today.strftime('%Y-%m-%d')
        #results_death.insert(0, 'model', 'UT-GGPNBLDS')
        results_death.insert(0, 'forecast_date', today_date)
        results_death_weekly.insert(0, 'forecast_date', today_date)
        results_daily_cases_weekly.insert(0, 'forecast_date', today_date)
        #death_mean['target'] = today
        #death_mean['forecase_date'] = pd.to_datetime(death_mean.forecast_date).dt.strftime('%Y-%m-%d')
        #start_date = death_mean['forecast_date'].loc[1]
        #date_1 = datetime.datetime.strptime(start_date, "%y-%m-%d")
        #end_date = date_1 + datetime.timedelta(days=10)
        for i in range(4):
            if i==0:
                temp = today+ datetime.timedelta(days=5)
                temp_date = temp.strftime('%Y-%m-%d')
                
            else:
                temp = today + datetime.timedelta(days=5+i*7)
                temp_date =  temp.strftime('%Y-%m-%d')
                
            temp_df = results_death.loc[results_death['date'] == temp_date]
            temp_df.insert(2,'target',str(i+1)+' wk ahead cum death')
            
           
                
            temp_df_weekly = results_death_weekly.loc[results_death_weekly['date'] == temp_date]
            
            
            temp_df_cases_weekly = results_daily_cases_weekly.loc[results_daily_cases_weekly['date'] == temp_date]
            
            
            if i==0:
                sunday_date = today + datetime.timedelta(days= -1)
                sunday_date = sunday_date.strftime('%Y-%m-%d')
                sunday_results_death = results_death_weekly.loc[results_death_weekly['date'] == sunday_date]
                sunday_results_cases = results_daily_cases_weekly.loc[results_death_weekly['date'] == sunday_date]
                temp_df_weekly['number_of_deaths'] = temp_df_weekly['number_of_deaths'] + sunday_results_death['number_of_deaths'].values
                temp_df_cases_weekly['number_of_deaths'] = temp_df_cases_weekly['number_of_deaths'] + sunday_results_cases['number_of_deaths'].values
                
            
            temp_df_weekly.insert(2,'target',str(i+1)+' wk ahead inc death')
            temp_df_cases_weekly.insert(2,'target',str(i+1)+' wk ahead inc case')
                
            if i ==0:
                cdc_df = temp_df
                cdc_df = cdc_df.append(temp_df_weekly)
                cdc_df = cdc_df.append(temp_df_cases_weekly)
                
            else:
                cdc_df = cdc_df.append(temp_df)
                cdc_df = cdc_df.append(temp_df_weekly)
                cdc_df = cdc_df.append(temp_df_cases_weekly)
    

    #cdc_df = cdc_df.rename(columns={'Province_State': 'location_name', 'date': 'target_week_end_date','number_of_deaths': 'point','number_of_deaths_lower': '0.025' ,'number_of_deaths_higher': '0.975'})      
    temp = cdc_df['type'].loc[cdc_df['type'] !='point']
    cdc_df['type'].loc[cdc_df['type'] !='point']='quantile'
    cdc_df.insert(4, 'quantile', 'NA')
    
    #cdc_df.insert(1, 'forecast_date', today_date)
    cdc_df['quantile'].loc[cdc_df['type'] !='point']=temp
    cdc_df = cdc_df.replace({"Province_State": us_state_code}) 
    #cdc_df[]
    cdc_df = cdc_df.rename(columns={'Province_State': 'location', 'date': 'target_end_date','number_of_deaths': 'value','number_of_deaths_lower': '0.025' ,'number_of_deaths_higher': '0.975'})      
cdc_df.to_csv('results/'+foldername+'_prediction/cdc_cum_death_cases.csv', index = False, header=True)           
            


    
    

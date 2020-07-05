import matlab.engine
import pandas as pd
import numpy as np
from datetime import datetime as dt

# model_name = 'DPGDS' # 'DPGDS' 'DPFA'
# # process and save the covid data
# df = (pd.read_csv("data/time_series_covid19_deaths_US.csv"))
# df1 = df.groupby(['Province_State']).sum()
# df1 = df1.loc[:,'1/22/20':]
# df1 = df1.drop(index=['Guam','Virgin Islands','Puerto Rico','Grand Princess','American Samoa','Northern Mariana Islands','Diamond Princess'])
# new_death_cases = df1 - df1.shift(1,axis=1)
# new_death_cases[new_death_cases<0]=0
# df = pd.read_csv("data/time_series_covid19_confirmed_US.csv")
# df2 = df.groupby(['Province_State']).sum()
# df2 = df2.loc[:,'1/22/20':]
# df2 = df2.drop(index=['Guam','Virgin Islands','Puerto Rico','Grand Princess','American Samoa','Northern Mariana Islands','Diamond Princess'])
# new_daily_cases = df2 - df2.shift(1,axis=1)
# new_daily_cases[new_daily_cases<0]=0
# new_death_cases.to_csv ('data/new_death_cases.csv', index = True, header=True)
# new_daily_cases.to_csv ('data/new_daily_cases.csv', index = True, header=True)


# call the matlab code with matlab.engine
# engine=matlab.engine.start_matlab()
# if model_name=='DPGDS':
#     print('calling DPGDS matlab code')
#     engine.main_DPGDS(nargout=0)
# else:
#     print('calling DPFA matlab code')
#     engine.main_DPFA(nargout=0)
# #engine.main_DPFA(nargout=0)


#############################
save_file_histor= 'D:/covid2019/Covid_DPGDS/results/Historic_prediction/layer1/K1_15/'
save_file_future= 'D:/covid2019/Covid_DPGDS/results/Future_prediction/layer1/K1_15/'


death_mean = pd.read_csv(save_file_histor + 'death_mean.csv')
death_mean = death_mean.rename(columns=death_mean.loc[0,:])
death_mean=(death_mean.drop(index=0))
death_lower = pd.read_csv(save_file_histor + 'death_lowerBound.csv')
death_lower = death_lower.rename(columns=death_lower.loc[0,:])
death_lower = (death_lower.drop(index=0))
death_higher = pd.read_csv(save_file_histor +  'death_upperBound.csv')
death_higher = death_higher.rename(columns=death_higher.loc[0,:])
death_higher = (death_higher.drop(index=0))
for i in range(1,53):
    death_mean.loc[i,'3/15/20':] =  pd.to_numeric(death_mean.loc[i,'3/15/20':],errors='coerce')
    death_lower.loc[i,'3/15/20':] =  pd.to_numeric(death_lower.loc[i,'3/15/20':],errors='coerce')
    death_higher.loc[i,'3/15/20':] =  pd.to_numeric(death_higher.loc[i,'3/15/20':],errors='coerce')



death_mean.to_csv (save_file_histor+ 'death_mean_.csv', index = True, header=True)
death_lower.to_csv (save_file_histor+'death_lowerBound_.csv', index = True, header=True)
death_higher.to_csv (save_file_histor+'death_upperBound_.csv', index = True, header=True)


death_mean_col = pd.melt(death_mean, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths')
death_lower_col = pd.melt(death_lower, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_lower')
death_higher_col = pd.melt(death_higher, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_higher')
results_death= pd.merge(death_mean_col, death_lower_col, how='outer', on=['Province_State', 'date'])
results_death= pd.merge(results_death, death_higher_col, how='outer', on=['Province_State', 'date'])
results_death.to_csv (save_file_histor +'historic_death_.csv', index = True, header=True)
#################################################################################################


death_mean = pd.read_csv(save_file_future+ 'death_mean.csv')
death_mean = death_mean.rename(columns=death_mean.loc[0,:])
death_mean=(death_mean.drop(index=0))
death_lower = pd.read_csv(save_file_future+'death_lowerBound.csv')
death_lower = death_lower.rename(columns=death_lower.loc[0,:])
death_lower = (death_lower.drop(index=0))
death_higher = pd.read_csv(save_file_future + 'death_upperBound.csv')
death_higher = death_higher.rename(columns=death_higher.loc[0,:])
death_higher = (death_higher.drop(index=0))
for i in range(1,53):
    death_mean.loc[i,'3/15/20':] =  pd.to_numeric(death_mean.loc[i,'3/15/20':],errors='coerce')
    death_lower.loc[i,'3/15/20':] =  pd.to_numeric(death_lower.loc[i,'3/15/20':],errors='coerce')
    death_higher.loc[i,'3/15/20':] =  pd.to_numeric(death_higher.loc[i,'3/15/20':],errors='coerce')

death_mean.to_csv (save_file_future+'death_mean_.csv', index = True, header=True)
death_lower.to_csv (save_file_future+'death_lowerBound_.csv', index = True, header=True)
death_higher.to_csv (save_file_future+'death_upperBound_.csv', index = True, header=True)



###########################################################################################
death_mean_col = pd.melt(death_mean, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths')
death_lower_col = pd.melt(death_lower, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_lower')
death_higher_col = pd.melt(death_higher, id_vars=['Province_State'], var_name='date', value_name='number_of_deaths_higher')
results_death= pd.merge(death_mean_col, death_lower_col, how='outer', on=['Province_State', 'date'])
results_death= pd.merge(results_death, death_higher_col, how='outer', on=['Province_State', 'date'])



results_death.to_csv (save_file_future+'furture_death_.csv', index = True, header=True)


###########################################################################################



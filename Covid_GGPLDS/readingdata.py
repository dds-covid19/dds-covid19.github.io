#!/usr/bin/env python2
# -*- coding: utf-8 -*-

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
last_day_cum_death = df1.iloc[:,-1]
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
df2 = df.groupby(['Province_State']).sum()
df2 = df2.loc[:,'1/22/20':]
df2 = df2.drop(index=['Guam','Virgin Islands','Puerto Rico','Grand Princess','American Samoa','Northern Mariana Islands','Diamond Princess'])
new_daily_cases = df2 - df2.shift(1,axis=1)
new_daily_cases[new_daily_cases<0]=0
new_death_cases.to_csv ('data/new_death_cases_2020_08_16.csv', index = True, header=True)
new_daily_cases.to_csv ('data/new_daily_cases_2020_08_16.csv', index = True, header=True)
last_day_cum_death.to_csv('data/last_cum_death_cases.csv', index = True, header=True)

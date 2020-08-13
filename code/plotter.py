#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 00:17:51 2020

@author: rahikalantari
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:24:46 2020

@author: rahikalantari
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np 

task = 'future' #'historic''future'
event = 'cases'#'cases''death'
if task == 'historic':
    foldername = 'Historic'
else:
    foldername = 'Future'

if event == 'death':
    sfilename = 'death'
else:
    sfilename = 'daily_cases'

if event == 'death':
    
    death_ = pd.read_csv('../Covid_GGPLDS/results/'+foldername+'_prediction/'+foldername+'__'+sfilename+'_2020_08_10.csv').drop('Unnamed: 0', axis=1)
    state_names = np.unique(death_['Province_State'])
    temp=np.append(state_names[44],state_names[0:44])
    state_names=np.append(temp,state_names[45:])
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    fig = go.Figure()
    for i in range (52):
        temp = death_.loc[death_['Province_State'] == state_names[i]]
        x_prediction = temp['date'].iloc[-30:]
        y_prediction = temp['number_of_deaths'].iloc[-30:]

        y_lower = temp['number_of_deaths_lower'].iloc[-30:]
        y_higher = temp['number_of_deaths_higher'].iloc[-30:]
       
        x_fit = temp['date'].iloc[52:-30]
        y_fit = temp['number_of_deaths'].iloc[52:-30]

        
        x_real = temp['date']
        y_real = temp['real_number_of_deaths']

    
    
        fig.add_trace(go.Scatter(
        x=x_real,
        y=y_real,
        name= state_names[i]+' Real data',
    ))
    

   
        if i<=51:

         
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                name = state_names[i]+' fit data', # Style name/legend entry with html tags
                connectgaps=True # override default to connect the gaps
            ))

        fig.add_trace(go.Scatter(
            x=x_prediction,
            y=y_prediction,
            name = state_names[i]+ ' forecase mean', # Style name/legend entry with html tags
            connectgaps=True # override default to connect the gaps
        ))
        
        fig.add_trace(go.Scatter(
            x=x_prediction,
            y=y_lower,
            mode='lines',
            line=dict(width=0.5, color='rgb(131, 90, 241)'),
            #stackgroup='one',
            name = state_names[i] + ' forecast 2.5 th precentile', # Style name/legend entry with html tags
            connectgaps=True # override default to connect the gaps
        ))
        
        fig.add_trace(go.Scatter(
            x=x_prediction,
            y=y_higher,
            hoverinfo='x+y',
            mode='lines',
            line=dict(width=0.5, color='rgb(131, 90, 241)'),
            fill='tonexty',
            #stackgroup='one', # define stack group
            name = state_names[i]+' forecast 97.5 th precentile', # Style name/legend entry with html tags
            connectgaps=True # override default to connect the gaps
        ))
        
else:

    cases_ = pd.read_csv('../Covid_GGPLDS/results/'+foldername+'_prediction/'+foldername+'__daily_cases_2020_08_10.csv').drop('Unnamed: 0', axis=1)
    state_names = np.unique(cases_['Province_State'])
    temp=np.append(state_names[44],state_names[0:44])
    state_names=np.append(temp,state_names[45:])
    

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    fig = go.Figure()
    for i in range (52):
        temp = cases_.loc[cases_['Province_State'] == state_names[i]]
        x_prediction = temp['date'].iloc[-30:]
        y_prediction = temp['number_of_daily_cases'].iloc[-30:]
        y_lower = temp['number_of_daily_cases_lower'].iloc[-30:]
        y_higher = temp['number_of_daily_cases_higher'].iloc[-30:]
       
        x_fit = temp['date'].iloc[52:-30]
        y_fit = temp['number_of_daily_cases'].iloc[52:-30]
        
        x_real = temp['date']
        y_real = temp['real_number_of_daily_cases']

    
        fig.add_trace(go.Scatter(
        x=x_real,
        y=y_real,
        name= state_names[i]+' Real data',
        ))
    

        if i<=51:
         
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                name = state_names[i]+' fit data', # Style name/legend entry with html tags
                connectgaps=True # override default to connect the gaps
            ))
         
        fig.add_trace(go.Scatter(
            x=x_prediction,
            y=y_prediction,
            name = state_names[i]+ ' forecase mean', # Style name/legend entry with html tags
            connectgaps=True # override default to connect the gaps
        ))
        
        fig.add_trace(go.Scatter(
            x=x_prediction,
            y=y_lower,
            mode='lines',
            line=dict(width=0.5, color='rgb(131, 90, 241)'),
            #stackgroup='one',
            name = state_names[i] + ' forecast 2.5 th precentile', # Style name/legend entry with html tags
            connectgaps=True # override default to connect the gaps
        ))
        
        fig.add_trace(go.Scatter(
            x=x_prediction,
            y=y_higher,
            hoverinfo='x+y',
            mode='lines',
            line=dict(width=0.5, color='rgb(131, 90, 241)'),
            fill='tonexty',
            #stackgroup='one', # define stack group
            name = state_names[i]+' forecast 97.5 th precentile', # Style name/legend entry with html tags
            connectgaps=True # override default to connect the gaps
        ))
        
        
list_updatemenus = []
for i in np.arange(52):
    visible = np.full(52*5,False)
    visible[i*5:(i+1)*5] = True
    temp_dict = dict(label = state_names[i],
                 method = 'update',
                 args = [{'visible': visible},
                         {'title': state_names[i]}])
    list_updatemenus.append(temp_dict)

fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=list_updatemenus,
        )
    ])
   

fig.show()
fig.write_html('../images/fig_'+foldername+'_'+sfilename+'.html')


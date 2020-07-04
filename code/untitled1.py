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


death_historic = pd.read_csv('../Covid_GGPLDS/results/Historic_prediction//historic_death_.csv').drop('Unnamed: 0', axis=1)
state_names = np.unique(death_historic['Province_State'])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
fig = go.Figure()
for i in range (52):
    temp = death_historic.loc[death_historic['Province_State'] == state_names[i]]
    x_prediction = temp['date'].iloc[-7:]
    y_prediction = temp['number_of_deaths'].iloc[-7:]
    y_lower = temp['number_of_deaths_lower'].iloc[-7:]
    y_higher = temp['number_of_deaths_higher'].iloc[-7:]
   
    x_fit = temp['date'].iloc[52:-7]
    y_fit = temp['number_of_deaths_higher'].iloc[52:-7]
    
    x_real = temp['date']
    y_real = temp['number_of_deaths']

    
    
    fig.add_trace(go.Scatter(
        x=x_real,
        y=y_real,
        name= state_names[i]+' Real data',
    ))
     
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        name = state_names[i]+' fit data', # Style name/legend entry with html tags
        connectgaps=True # override default to connect the gaps
    ))
     
    fig.add_trace(go.Scatter(
        x=x_prediction,
        y=y_prediction,
        name = state_names[i]+ ' mean of predictions', # Style name/legend entry with html tags
        connectgaps=True # override default to connect the gaps
    ))
    
    fig.add_trace(go.Scatter(
        x=x_prediction,
        y=y_lower,
        mode='lines',
        line=dict(width=0.5, color='rgb(131, 90, 241)'),
        name = state_names[i] + ' Lowe bound', # Style name/legend entry with html tags
        connectgaps=True # override default to connect the gaps
    ))
    
    fig.add_trace(go.Scatter(
        x=x_prediction,
        y=y_higher,
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5, color='rgb(131, 90, 241)'),
        stackgroup='one', # define stack group
        name = state_names[i]+' Upper bound', # Style name/legend entry with html tags
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
fig.write_html("../images/fig11.html")


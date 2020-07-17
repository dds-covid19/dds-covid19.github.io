#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:53:02 2020

@author: zhoum
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

df = pd.read_csv('../data/new_death_cases.csv')
del df['1/22/20']
df.loc[51] = df.sum(axis=0)
df["Province_State"].loc[51] = "US"
US =df.loc[51][1:]


State_Names = df['Province_State'].tolist()


Total = pd.DataFrame(data=US)
Total['Date']=Total.index
Total = Total.rename(columns={51: "Number of Deaths"})

fig = go.Figure()

for i in np.arange(52):
    
    state =df.loc[i][1:]
    state = pd.DataFrame(data=state)
    state = state.rename(columns={i: "Number of Deaths"})
    
    fig.add_trace(
        go.Scatter(
            x = state.index,
            y = state["Number of Deaths"],
            name = State_Names[i]
        )
    )

#drop=np.array[]
#for i in np.arange(52):
#    visible = np.full(52,False)
#    visible[i]=True
#    a=dict(label = 'All',
#            method = 'update',
#            args = [{'visible': visible},
#                  {'title': State_Names[i],
#                   'showlegend':True}])
#    if i==0:
#        drop=list(a)
#    else:
#        drop.append(a)
        
        
list_updatemenus = []
for i in np.arange(52):
    visible = np.full(52,False)
    visible[i] = True
    temp_dict = dict(label = State_Names[i],
                 method = 'update',
                 args = [{'visible': visible},
                         {'title': State_Names[i]}])
    list_updatemenus.append(temp_dict)
    

    
fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=list_updatemenus,
        )
    ])

        
#fig.show()            
fig.write_html("../images/fig.html")

#if __name__ == '__main__':
#    app.run_server(debug=True)
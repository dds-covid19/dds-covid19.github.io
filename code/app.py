#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:53:02 2020

@author: zhoum
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "wide-form" data frame with no index
# see https://plotly.com/python/wide-form/ for more options
df = pd.DataFrame({"x": [1, 2, 3], "SF": [4, 1, 2], "Montreal": [2, 4, 5]})

df = pd.read_csv('../data/new_death_cases.csv')
US=df.sum()
US=US[2:]


del df['1/22/20']
df.loc[51] = df.sum(axis=0)
df["Province_State"].loc[51] = "US"
US =df.loc[51][1:]
#US.plot()
#plt.show()



Total = pd.DataFrame(data=US)
Total['Date']=Total.index
Total = Total.rename(columns={51: "Number of Deaths"})

#df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')


#def generate_table(dataframe, max_rows=20):
#    return html.Table([
#        html.Thead(
#            html.Tr([html.Th(col) for col in dataframe.columns])
#        ),
#        html.Tbody([
#            html.Tr([
#                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#            ]) for i in range(min(len(dataframe), max_rows))
#        ])
#    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#df1 = pd.DataFrame({"x": np.sum(df), "SF": [4, 1, 2], "Montreal": [2, 4, 5]})

fig = px.line(Total, x="Date", y="Number of Deaths")
fig.write_html("../images/fig.html")

#
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
#app.layout = html.Div(children=[
#    html.H4(children='COVID-19 deaths in the U.S.'),
#    #generate_table(df)
#    dcc.Graph(
#        id='example-graph',
#        figure=fig
#    )
#])
#    
#    
#
#if __name__ == '__main__':
#    #app.run_server(debug=True)
#    app.run_server()
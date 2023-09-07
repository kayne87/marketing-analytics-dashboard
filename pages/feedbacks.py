import dash
from dash import dcc, html, callback, Output, Input, clientside_callback
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from datetime import date
from datetime import datetime
import pandas as pd
import numpy as np
import os 

from collections import Counter
from wordcloud import WordCloud
import base64
import io
from PIL import Image

dash.register_page(__name__, name='Feedbacks', order=4)
cherrypicked_seed = 42

path = ".//data//"
df_sentiments = pd.read_csv(path + "customers_sentiments.csv")
df_rfm = pd.read_csv(path + "customers_rfm_jenks.csv").round(2)
df_orders = pd.read_csv(path + "orders.csv")

df_orders["purchase_datetime"] = pd.to_datetime(df_orders["purchase_datetime"])
print(len(df_orders))

df_orders_active = df_orders[df_orders["purchase_datetime"] >= '2023-01-01']

df_sentiments.loc[df_sentiments["sentiment"] == 1, "sentiment_description"] = "Positive"
df_sentiments.loc[df_sentiments["sentiment"] == 0, "sentiment_description"] = "Neutral"
df_sentiments.loc[df_sentiments["sentiment"] == -1, "sentiment_description"] = "Negative"

df_sentiments_pie = df_sentiments.groupby(by=["sentiment_description"]).count().reset_index()[["sentiment_description", "sentiment"]]

total = df_sentiments_pie["sentiment"].sum()
positive = df_sentiments_pie[df_sentiments_pie["sentiment_description"] == "Positive"]["sentiment"].values[0]
negative = df_sentiments_pie[df_sentiments_pie["sentiment_description"] == "Negative"]["sentiment"].values[0]

brs = (positive/total) - (negative/total)
brs = round(brs*100, 2)

df_sentiments_active = pd.merge(
    df_sentiments,
    df_orders_active,
    on="customer_id", 
    how='inner'
)
active_users_sentiments = df_sentiments_active["customer_id"].unique()
df_sentiments_active = df_sentiments[df_sentiments["customer_id"].isin(active_users_sentiments)]

df_sentiments_active_pie = df_sentiments_active.groupby(by=["sentiment_description"]).count().reset_index()[["sentiment_description", "sentiment"]]

layout = html.Div(
    [
        dcc.Markdown('# Customer satisfaction'),
        html.Div(children=
            [
                html.Div(children=
                    [
                        html.P('What do customers talk about', style={'font-weight': 'bold'}),
                        html.P('Customer reviews (top 100 reviews)', style={'font-weight': 'bold', 'color': 'gray'}),
                        html.Img(
                            src="/assets/customer_sentiments.png",
                            alt="Studied reviews",
                            width=400
                        ),
                        html.P('.', style={'color': 'transparent'}),
                        html.P('Machine learning training reviews (top 100 reviews)', style={'font-weight': 'bold', 'color': 'gray'}),
                        html.Img(
                            src="/assets/training_reviews.png",
                            alt="Studied reviews",
                            width=400
                        ),
                    ], style={'padding': 10, 'flex-basis': '420px', 'flex-grow': 0, 'flex-shrink': 0}
                ),

                html.Div(children=
                    [
                        html.P('Predicted BRS (Brand Reccomandation Score): %' + str(brs) + '%', style={'font-weight': 'bold'}),
                        html.P('Predicted sentiments', style={'font-weight': 'bold', 'color': 'gray'}),
                        dcc.Graph(
                            id='sentiment_pie',
                            figure = px.pie(
                                df_sentiments_pie,
                                values='sentiment',
                                names='sentiment_description',
                                color='sentiment_description',
                                color_discrete_map={ 'Positive': '#8DC2B8', 'Neutral': '#e9e5e5', 'Negative': 'black'}
                            )
                        )
                    ], style={'padding': 10, 'flex': 1}
                ),

                html.Div(children=
                    [
                        html.P('Active users in Q1 2023', style={'font-weight': 'bold', 'color': 'gray'}),
                        dcc.Graph(
                            id='sentiment_active_pie',
                            figure = px.pie(
                                df_sentiments_active_pie,
                                values='sentiment',
                                names='sentiment_description',
                                color='sentiment_description',
                                color_discrete_map={ 'Positive': '#8DC2B8', 'Neutral': '#e9e5e5', 'Negative': 'black'}
                            )
                        )
                    ], style={'padding': 10, 'flex': 1}
                )
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        ),

        dcc.Interval(
            id="interval-component",
            interval=1000,  # Interval in milliseconds (1 second)
            n_intervals=0
        ),
    ]
)
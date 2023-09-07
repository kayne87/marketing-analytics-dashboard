import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_mantine_components as dmc
import dash_ag_grid as dag
import geopandas as gpd

from datetime import date
from datetime import datetime
import pandas as pd
import numpy as np
import os 

dash.register_page(__name__, path='/', name='Revenue', order=0) # '/' is home page
path = ".//data//"

# LOAD DATA

df_orders = pd.read_csv(path + "orders.csv")
df_orders["purchase_datetime"] = pd.to_datetime(df_orders["purchase_datetime"])
df_customers = pd.read_csv(path + "customers.csv")

dt_min = df_orders["purchase_datetime"].min()
dt_max = df_orders["purchase_datetime"].max()

df_customers_orders = pd.merge(left=df_orders, right=df_customers, how='left', left_on='customer_id', right_on='customer_id')
df_regions_stat_gross = df_customers_orders[["region", "gross_price"]].groupby(by=["region"]).sum()
df_regions_stat_count = df_customers_orders[["order_id", "region"]].drop_duplicates().groupby(by=["region"]).count()
df_regions_stats = pd.merge(left=df_regions_stat_gross, right=df_regions_stat_count, how='left', left_on='region', right_on='region')

gdf = gpd.read_file('../data/geospatial/limits_IT_regions.geojson')
gdf = gdf.reset_index()
gdf.iloc[0, gdf.columns.get_loc("reg_name")] = "PIEMONTE"
gdf.iloc[1, gdf.columns.get_loc("reg_name")] = "VALLE D'AOSTA"
gdf.iloc[2, gdf.columns.get_loc("reg_name")] = "LOMBARDIA"
gdf.iloc[3, gdf.columns.get_loc("reg_name")] = "TRENTINO ALTO ADIGE"
gdf.iloc[4, gdf.columns.get_loc("reg_name")] = "VENETO"
gdf.iloc[5, gdf.columns.get_loc("reg_name")] = "FRIULI VENEZIA GIULIA"
gdf.iloc[6, gdf.columns.get_loc("reg_name")] = "LIGURIA"
gdf.iloc[7, gdf.columns.get_loc("reg_name")] = "EMILIA ROMAGNA"
gdf.iloc[8, gdf.columns.get_loc("reg_name")] = "TOSCANA"
gdf.iloc[9, gdf.columns.get_loc("reg_name")] = "UMBRIA"
gdf.iloc[10, gdf.columns.get_loc("reg_name")] = "MARCHE"
gdf.iloc[11, gdf.columns.get_loc("reg_name")] = "LAZIO"
gdf.iloc[12, gdf.columns.get_loc("reg_name")] = "ABRUZZO"
gdf.iloc[13, gdf.columns.get_loc("reg_name")] = "MOLISE"
gdf.iloc[14, gdf.columns.get_loc("reg_name")] = "CAMPANIA"
gdf.iloc[15, gdf.columns.get_loc("reg_name")] = "PUGLIA"
gdf.iloc[16, gdf.columns.get_loc("reg_name")] = "BASILICATA"
gdf.iloc[17, gdf.columns.get_loc("reg_name")] = "CALABRIA"
gdf.iloc[18, gdf.columns.get_loc("reg_name")] = "SICILIA"
gdf.iloc[19, gdf.columns.get_loc("reg_name")] = "SARDEGNA"

gdf_stats = pd.merge(left=gdf, right=df_regions_stats, how='left', left_on='reg_name', right_on='region')
bbox = gdf_stats.geometry.total_bounds
center_lat = (bbox[1] + bbox[3]) / 2
center_lon = (bbox[0] + bbox[2]) / 2

df_customers_orders["QUARTER"] = ""
df_customers_orders.loc[df_customers_orders["purchase_datetime"] >= '2022-05-01', "QUARTER"] = "Q2 2022"
df_customers_orders.loc[df_customers_orders["purchase_datetime"] >= '2022-09-01', "QUARTER"] = "Q3 2022"
df_customers_orders.loc[df_customers_orders["purchase_datetime"] >= '2023-01-01', "QUARTER"] = "Q1 2023"

df_orders_quarter = df_customers_orders[["QUARTER", "gross_price"]].groupby(by=["QUARTER"]).sum().reset_index()
df_orders_gender = df_customers_orders[["gender", "gross_price"]].groupby(by=["gender"]).sum().reset_index()
df_orders_age = df_customers_orders[["age", "gross_price"]].groupby(by=["age"]).sum().reset_index()

df_orders_month = df_customers_orders.copy()
df_orders_month['month_year'] = df_orders_month['purchase_datetime'].dt.strftime('%Y-%m')
df_orders_month = df_orders_month.groupby('month_year')['gross_price'].sum().reset_index()[["month_year", "gross_price"]]
df_orders_month = df_orders_month.sort_values(by=["month_year"])

df_orders_quarter.loc[df_orders_quarter["QUARTER"] == 'Q2 2022', "POS"] = 1
df_orders_quarter.loc[df_orders_quarter["QUARTER"] == 'Q3 2022', "POS"] = 2
df_orders_quarter.loc[df_orders_quarter["QUARTER"] == 'Q1 2023', "POS"] = 3

df_orders_quarter = df_orders_quarter.sort_values(by=["POS"])

ages_plus = [i * 10 for i in range(2, 8)]
for age in ages_plus:
    df_orders_age.loc[df_orders_age["age"] >= (age if age >= 20 else 18), "AGE+"] = str(age) + "+"

df_orders_age = df_orders_age[["AGE+", "gross_price"]].groupby(by=["AGE+"]).sum().reset_index()
df_orders_age = df_orders_age.sort_values(by=["AGE+"])

def format_in_millions(x):
    return f'{x / 1e6:.2f}M'  # Divide by 1 million and format with 2 decimal places and "M"

layout = html.Div(
    [
        dcc.Markdown('# Some contextual analysis'),
        html.Div(children=
            [
                html.Div(children=
                    [
                        html.P('Gross revenue by regions', style={'font-weight': 'bold'}),
                        dcc.Graph(
                            id='italy-map-revenue',
                            figure = px.choropleth(
                                gdf_stats.round(2),
                                color_continuous_scale='mint',
                                hover_name="reg_name",
                                geojson=gdf.geometry,
                                locations=gdf.index,
                                color='gross_price',
                                scope="europe",
                            ).update_geos(
                                center=dict(lon=12.5, lat=41.9),
                                projection_scale=6,
                                visible=False  # Hide globe background
                            ).update_traces(showlegend=False).update_layout(
                                margin={"t": 0, "r": 0, "b": 0, "l": 0}, 
                                height=400, 
                                autosize=True
                            ),
                            config = {'displayModeBar': False}
                        )
                    ], style={'padding': 10, 'flex-basis': '500px', 'flex-grow': 0, 'flex-shrink': 0}
                ),

                html.Div(children=
                    [
                        html.P('Gross revenue by age', style={'font-weight': 'bold'}),
                        dcc.Graph(
                            id='gross-revenues-age',
                            figure=px.bar(
                                df_orders_age, 
                                x='AGE+', 
                                y='gross_price',
                                text=df_orders_age['gross_price'].apply(format_in_millions)
                            ).update_traces(marker_color='#8DC2B8',showlegend=False,textfont=dict(size=14, family='Arial')).update_layout(
                                margin={"t": 0, "r": 0, "b": 0, "l": 0}, 
                                height=400, 
                                autosize=True
                            ),
                            config = {'displayModeBar': False}
                        )
                    ], style={'padding': 10, 'flex': 1}
                )
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        ),

        html.Div(children=
            [
                html.Div(children=
                    [
                        html.P('Gross revenue by gender', style={'font-weight': 'bold'}),
                        dcc.Graph(
                            id='gross-revenue-gender',
                            figure = px.pie(
                                df_orders_gender,
                                values='gross_price',
                                names='gender',
                                color='gender',
                                color_discrete_map={ 'M': '#8AC2F7', 'F': '#ffa9d0'}
                            )
                        )
                    ], style={'padding': 10, 'flex-basis': '500px', 'flex-grow': 0, 'flex-shrink': 0}
                ),

                html.Div(children=
                    [
                        html.P('Revenue vs Quarters', style={'font-weight': 'bold'}),
                        dcc.Graph(
                            id='revenues-quarters',
                            figure=px.bar(
                                df_orders_quarter, 
                                x='QUARTER', 
                                y='gross_price',
                                text=df_orders_quarter['gross_price'].apply(format_in_millions)
                            ).update_traces(marker_color='#8DC2B8',showlegend=False,textfont=dict(size=14, family='Arial'),textposition='outside').update_layout(
                                margin={"t": 0, "r": 0, "b": 0, "l": 0}, 
                                height=400, 
                                autosize=True,
                                yaxis=dict(range=[4500000, 7000000])
                            ),
                            config = {'displayModeBar': False}
                        )
                    ], style={'padding': 10, 'flex': 1}
                )
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        ),

        html.Div(children=
            [
                html.P('Gross revenue by months', style={'font-weight': 'bold'}),
                dcc.Graph(
                    id='gross-revenues-months',
                    figure=px.bar(
                        df_orders_month, 
                        x='month_year', 
                        y='gross_price',
                        text=df_orders_month['gross_price'].apply(format_in_millions)
                    ).update_traces(marker_color='#8DC2B8',showlegend=False,textfont=dict(size=14, family='Arial')).update_layout(
                        margin={"t": 0, "r": 0, "b": 0, "l": 0}, 
                        height=400, 
                        autosize=True
                    ),
                    config = {'displayModeBar': False}
                )
            ], style={'padding': 10, 'flex': 1}
        ),

        html.P('.', style={'color': 'transparent', 'font-weight': 'bold'}),
        html.P('.', style={'color': 'transparent', 'font-weight': 'bold'}),
        html.P('.', style={'color': 'transparent', 'font-weight': 'bold'}),

        dcc.Interval(
            id="interval-component",
            interval=1000,  # Interval in milliseconds (1 second)
            n_intervals=0
        ),
    ]
)

"""
        ,
        """

"""
@callback(
    Output('segment-users-churn', 'rowData'),
    [Input('filter-segment-churn', 'value')]
)
def update_table(selected_segment):
    df_return = df_churner_table[df_churner_table["CLASS"] == selected_segment]
    return df_return.to_dict("records")
"""
import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_mantine_components as dmc
import dash_ag_grid as dag

from datetime import date
from datetime import datetime
import pandas as pd
import numpy as np
import os 

dash.register_page(__name__, path='/churn', name='Customer churn', order=2) # '/' is home page
path = ".//data//"

def search_90percent_ecdf(ecdf):
    for i in range(0,len(ecdf)):
        if(ecdf[i] >= 90):
            return [i, ecdf[i]]

def compute_ecdf(distribution):
    counts = np.bincount(distribution)
    cumulative = np.cumsum(counts)
    bins = np.arange(len(cumulative))
    ecdf = (cumulative / len(distribution)) * 100
    return ecdf

def repurchase_curve(customer_average_periods):
    n_max = int(customer_average_periods.max())
    x = [i for i in range(0, n_max+1)]

    ecdf_repurchase = compute_ecdf(customer_average_periods)
    day90, cumulative90 = search_90percent_ecdf(ecdf_repurchase)

    tick_interval = 25
    x_ticks = np.sort(np.concatenate(
        (
            [1, day90], 
            [i for i in np.arange(25, max(x) + 1, tick_interval)]
        )
    ))

    y_ticks = [i for i in np.arange(0, 101, 10)]

    return [x_ticks, y_ticks]

# LOAD DATA

REFERENCE_DATE = "2023-01-01"

df_orders = pd.read_csv(path + "orders.csv")
df_orders_tbs = pd.read_csv(path + "orders_tbs.csv")
df_churners = pd.read_csv(path + "churners.csv")
df_orders["purchase_datetime"] = pd.to_datetime(df_orders["purchase_datetime"])

dt_min = df_orders["purchase_datetime"].min()
dt_max = df_orders["purchase_datetime"].max()

customers_prev = df_orders[
    (df_orders["purchase_datetime"] > "2022-09-01")
    & (df_orders["purchase_datetime"] < REFERENCE_DATE)
]["customer_id"].unique()

customers_mantained = df_orders[
    (df_orders["purchase_datetime"] >= REFERENCE_DATE)
    & (df_orders["customer_id"].isin(customers_prev))
]["customer_id"].unique()

active_customers = df_orders[
    (df_orders["purchase_datetime"] >= REFERENCE_DATE)
]["customer_id"].unique()

non_churners = round(len(customers_mantained)/len(customers_prev), 2)
churners = 1 - non_churners

churner_data = {
    'Category': ['CHURN', 'NON-CHURN'],
    'Values': [churners*100, non_churners*100]  # Adjust these counts as needed
}
df_churners_rate = pd.DataFrame(churner_data).round(2)

df_churner_table = df_churners[df_churners["CHURN"] == 1][["customer_id", "gender", "age", "tenure", "Recency", "Frequency", "Monetary", "CLASS"]].copy().round(2)

churn_color = "#FFBB5C"
layout = html.Div(
    [
        dcc.Markdown('# Churn analysis and risk prediction'),
        html.Div(children=
            [
                html.Div(children=
                    [
                        html.P('Churn rate in Q1 2023', style={'color': churn_color, 'font-weight': 'bold'}),
                        dcc.Graph(
                            id='scatter-m-f',
                            figure = px.pie(
                                df_churners_rate,
                                values='Values',
                                names='Category',
                                color='Category',
                                color_discrete_map={ 'CHURN': churn_color, 'NON-CHURN': '#4bbf73'}
                            ).update_traces(showlegend=False).update_layout(margin={"t": 0, "r": 0, "b": 0, "l": 0}, height=200, autosize=True),
                            config = {'displayModeBar': False}
                        )
                    ], style={'padding': 10, 'flex-basis': '230px', 'flex-grow': 0, 'flex-shrink': 0}
                ),
                html.Div(children=
                    [
                        dcc.Graph(
                            id='repurchase-curve',
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
                        html.P('Who will churn in Q2 2023?', style={'color': churn_color, 'font-weight': 'bold'}),
                        dmc.Timeline(
                            active=1,
                            bulletSize=15,
                            lineWidth=2,
                            children=[
                                dmc.TimelineItem(
                                    title="Q1 2023 - Jan/Apr",
                                    children=[
                                        dmc.Text(
                                            [
                                                "End of the known sales.",
                                            ],
                                            color="dimmed",
                                            size="sm",
                                            style={'font-weight':'bold'}
                                        ),
                                        dmc.Text(
                                            [
                                                "The active RFM highest valued customers are selected."
                                            ],
                                            color="dimmed",
                                            size="sm",
                                        ),
                                        dmc.Text(
                                            [
                                                "DIAMOND, GOLD, SILVER customers."
                                            ],
                                            color="dimmed",
                                            size="sm",
                                        )
                                    ],
                                ),
                                dmc.TimelineItem(
                                    title="Q2 2023 - May/Aug",
                                    children=[
                                        dmc.Text(
                                            [
                                                "The quarter period following the known one is the period in which we carry out the churn prediction",
                                            ],
                                            color="dimmed",
                                            size="sm",
                                        )
                                    ],
                                ),
                                dmc.TimelineItem(
                                    title="RISK OF CHURN",
                                    children=[
                                        dmc.Text(
                                            [
                                                "About 20% of high economic value customers are predicted to likely drop out",
                                            ],
                                            color="red",
                                            size="sm",
                                        )
                                    ],
                                )
                            ],
                        )
                    ], style={'padding': 10, 'flex-basis': '300px', 'flex-grow': 0, 'flex-shrink': 0}
                ),
                html.Div(children=
                    [
                        dcc.Dropdown(
                            id='filter-segment-churn',
                            options=[
                                {'label': 'DIAMOND', 'value': 'DIAMOND'},
                                {'label': 'GOLD', 'value': 'GOLD'},
                                {'label': 'SILVER', 'value': 'SILVER'},
                                {'label': 'BRONZE', 'value': 'BRONZE'},
                                {'label': 'COPPER', 'value': 'COPPER'},
                                {'label': 'TIN', 'value': 'TIN'},
                                {'label': 'CHEAP', 'value': 'CHEAP'}
                            ],
                            value='DIAMOND'
                        ),
                        dag.AgGrid(
                            id="segment-users-churn",
                            columnDefs= [
                                {"field": header} for header in df_churner_table.columns
                            ],
                            rowData= df_churner_table.to_dict('records'),
                            defaultColDef={"sortable": True, "filter": True},
                            className="ag-theme-balham",
                            columnSize="sizeToFit",
                            style={'height': '398px'},
                        )
                    ], style={'padding': 10, 'flex': 1}
                )
                
            ], style={'display': 'flex', 'flex-direction': 'row'}
        ),
        dcc.Interval(
            id="interval-component",
            interval=1000,  # Interval in milliseconds (1 second)
            n_intervals=0
        ),
    ]
)

@callback(
    Output('segment-users-churn', 'rowData'),
    [Input('filter-segment-churn', 'value')]
)
def update_table(selected_segment):
    df_return = df_churner_table[df_churner_table["CLASS"] == selected_segment]
    return df_return.to_dict("records")

@callback(
    Output('repurchase-curve', 'figure'),
    Input("interval-component", "n_intervals"))
def update_output(n_intervals):
    #COMPUTE ECDF PARAMS
    customer_average_days, x, y, day90, ecdf90, x_ticks, y_ticks = get_tbs_ecdf()

    figure = {
        'data': [
            go.Scatter(
                x = x,
                y = y,
                mode = 'lines',
                name = 'Distribuzione Cumulativa'
            ),
            go.Scatter(
                x = [day90, day90],
                y = [0, 100],
                mode='lines',
                name='-',
                line={'dash': 'dash', 'color': 'orange', 'width': 1}
            ),
            go.Scatter(
                x = [0, customer_average_days.max()],
                y = [ecdf90, ecdf90],
                mode='lines',
                name='-',
                line={'dash': 'dash', 'color': 'orange', 'width': 1}
            )
        ],
        'layout': go.Layout(
            xaxis = {'title': 'Repurchase average in days', 'tickvals': x_ticks},
            yaxis = {'title': 'Cumulative # users %', 'tickvals': y_ticks},
            margin={"t": 10, "r": 0, "b": 50, "l": 60}, height=300, autosize=True
        )
    }

    return figure

def get_tbs_ecdf():
    customer_average_periods = df_orders_tbs.groupby('customer_id')['TBS'].mean()
    n_max = int(customer_average_periods.max())
    x = [i for i in range(0, n_max+1)]

    ecdf_repurchase = compute_ecdf(customer_average_periods)
    day90, ecdf90 = search_90percent_ecdf(ecdf_repurchase)

    tick_interval = 25
    x_ticks = np.sort(np.concatenate(
        (
            [1, day90], 
            [i for i in np.arange(25, max(x) + 1, tick_interval)]
        )
    ))

    y_ticks = [i for i in np.arange(0, 101, 10)]
    return [customer_average_periods, x, ecdf_repurchase, day90, ecdf90, x_ticks, y_ticks]

"""
dcc.DatePickerRange(
            id = 'repurchase-curve-range',
            min_date_allowed = dt_min,
            max_date_allowed = dt_max,
            #initial_visible_month = dt_min,
            start_date = dt_min,
            end_date = dt_max,
            display_format = 'DD/MM/YYYY',
        ),
        dcc.Graph(
            id='repurchase-ecdf'
        )

@callback(
    Output('repurchase-ecdf', 'figure'),
    Input('repurchase-curve-range', 'start_date'),
    Input('repurchase-curve-range', 'end_date'))
def update_output(start_date, end_date):
    #FILTER THE DATAFRAME
    dt_start = datetime.strptime(start_date[:10], '%Y-%m-%d')
    dt_end = datetime.strptime(end_date[:10], '%Y-%m-%d')
    df_filtered = df_orders[(df_orders["purchase_datetime"] >= dt_start) & (df_orders["purchase_datetime"] <= dt_end)]
    #df_filtered = compute_tbs(df_filtered)
    
    #COMPUTE ECDF PARAMS
    customer_average_days, x, y, day90, ecdf90, x_ticks, y_ticks = get_tbs_ecdf(df_filtered)

    figure = {
        'data': [
            go.Scatter(
                x = x,
                y = y,
                mode = 'lines',
                name = 'Distribuzione Cumulativa'
            ),
            go.Scatter(
                x = [day90, day90],
                y = [0, 100],
                mode='lines',
                name='-',
                line={'dash': 'dash', 'color': 'orange', 'width': 1}
            ),
            go.Scatter(
                x = [0, customer_average_days.max()],
                y = [ecdf90, ecdf90],
                mode='lines',
                name='-',
                line={'dash': 'dash', 'color': 'orange', 'width': 1}
            )
        ],
        'layout': go.Layout(
            title = 'Cumulative Repurchase distribution %',
            xaxis = {'title': 'Repurchase average in days', 'tickvals': x_ticks},
            yaxis = {'title': 'Cumulative users %', 'tickvals': y_ticks}
        )
    }

    return figure


def get_tbs_ecdf(df_orders):
    #print(type(df_orders["purchase_datetime"][0].date()), type(dt_start))

    customer_average_periods = df_orders.groupby('customer_id')['TBS'].mean()
    print(len(customer_average_periods))
    n_max = int(customer_average_periods.max())
    x = [i for i in range(0, n_max+1)]

    counts = np.bincount(customer_average_periods)
    cumulative = np.cumsum(counts)
    bins = np.arange(len(cumulative))
    ecdf_repurchase = (cumulative / len(customer_average_periods)) * 100

    day90, ecdf90 = search_90percent_ecdf(ecdf_repurchase)

    tick_interval = 25

    x_ticks = np.sort(np.concatenate(
        (
            [1, day90], 
            [i for i in np.arange(25, max(x) + 1, tick_interval)]
        )
    ))

    y_ticks = [i for i in np.arange(0, 101, 10)]
    return [customer_average_periods, x, ecdf_repurchase, day90, ecdf90, x_ticks, y_ticks]

#
# Compute the TBS (Time Between Sales) for each customer with > 1 purchases
# <-- df_order: filtered dataframe where to compute the TBS field
# --> the resulting dataframe with the computed field 

def compute_tbs(df_orders):
    print(len(df_orders))
    
    df_orders = df_orders.sort_values(['customer_id', 'purchase_datetime'])
    df_orders['TBS'] = df_orders.groupby('customer_id')['purchase_datetime'].diff()
    df_orders['TBS'] = df_orders['TBS'].dt.days
    df_orders.loc[df_orders["TBS"].isna(), ["TBS"]] = 0
    df_orders = df_orders[df_orders["TBS"] > 0]

    print(len(df_orders))

    return df_orders

"""
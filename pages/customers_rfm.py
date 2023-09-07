import dash
from dash import dcc, html, callback, Output, Input, clientside_callback
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objs as go

from datetime import date
from datetime import datetime
import pandas as pd
import numpy as np
import os 

dash.register_page(__name__, path='/segments', name='Customer segments', order=1) # '/' is home page
path = ".//data//"

"""
LOADING FUNTIONS
"""

def get_rfm_stats():
    global df_rfm
    df_rfm_j_cid = df_rfm["customer_id"]
    df_orders_j = df_orders[df_orders["customer_id"].isin(df_rfm_j_cid)]
    df_order_j_avgorders = df_orders_j.groupby(["order_id", "customer_id"])[["product_id"]].count().reset_index().groupby(["customer_id"]).mean().reset_index().rename(columns={"product_id": "Products in ord."})

    df_rfm_stats = pd.concat([
        df_rfm.groupby(["CLASS"]).median()[["Recency", "Frequency", "Monetary"]].rename(columns={
            "Recency": "Recency (med.)", 
            "Frequency": "Frequency (med.)", 
            "Monetary": "Monetary (med.)"
        }).transpose()
        , pd.merge(
            df_order_j_avgorders, 
            df_rfm[["customer_id", "CLASS"]], 
            on="customer_id", 
            how='left'
        )[["Products in ord.", "CLASS"]].groupby(["CLASS"]).mean().transpose()
    ]).round(2)
    return df_rfm_stats

def get_rfm_monetary():
    global df_rfm
    df_grouped = df_rfm.groupby(["CLASS"]).median()[["Recency", "Frequency", "Monetary"]].sort_values(["Monetary"])
    return df_grouped.rename(columns = {'Monetary':'Monetary (med.)'}).reset_index()[["CLASS", "Monetary (med.)"]]
    #df_grouped[["AVG Monetary"]].plot.bar(ax=ax2, color="green")

def get_rfm_table():
    global df_rfm
    df_rfm = df_rfm[df_rfm["Monetary"] > 0]
    df_rfm_table = df_rfm[["customer_id", "Recency", "Frequency", "Monetary", "RFM_score", "CLASS"]].copy().rename(columns={"CLASS": "Segment"})
    return df_rfm_table

def get_rfm_totals():
    global df_rfm
    df_rfm_totals = df_rfm[["CLASS", "customer_id"]].groupby(["CLASS"]).count().reset_index().rename(columns={
        'CLASS': 'Segment',
        'customer_id': '# COUNT'
    })
    df_rfm_totals = df_rfm_totals.sort_values(['# COUNT'])
    return df_rfm_totals

"""
DATA PREPARE FOR THE DASHBOARD
"""

segs_map = { 'DIAMOND': '#63C5DA', 'GOLD': 'gold','SILVER': 'silver', 'BRONZE': '#cf7f32', 'COPPER': '#ED7D31', 'TIN': '#C00000', 'CHEAP': 'gray'}

df_orders = pd.read_csv(path + "orders.csv")
df_rfm = pd.read_csv(path + "customers_rfm_jenks.csv").round(2)

df_rfm_table = get_rfm_table()
df_rfm_totals = get_rfm_totals()
df_rfm_stats = get_rfm_stats()
df_rfm_monet = get_rfm_monetary()

cellStyle = {
    "styleConditions": [
        {
            "condition": "params.data.Segment == 'DIAMOND'",
            "style": {"backgroundColor": "#63C5DA", "color": "white", "fontWeight": "bold"},
        },
        {
            "condition": "params.data.Segment == 'GOLD'",
            "style": {"backgroundColor": "gold", "color": "white", "fontWeight": "bold"},
        },
        {
            "condition": "params.data.Segment == 'SILVER'",
            "style": {"backgroundColor": "silver", "color": "white", "fontWeight": "bold"},
        },
        {
            "condition": "params.data.Segment == 'BRONZE'",
            "style": {"backgroundColor": "#cf7f32", "color": "white", "fontWeight": "bold"},
        },
        {
            "condition": "params.data.Segment == 'COPPER'",
            "style": {"backgroundColor": "#ED7D31", "color": "white", "fontWeight": "bold"},
        },
        {
            "condition": "params.data.Segment == 'TIN'",
            "style": {"backgroundColor": "#C00000", "color": "white", "fontWeight": "bold"},
        },
        {
            "condition": "params.data.Segment == 'CHEAP'",
            "style": {"backgroundColor": "gray", "color": "white", "fontWeight": "bold"},
        }
    ]
}

"""
UI LAYOUT
"""

layout = html.Div(
    [

        dcc.Markdown('# RFM Analysis / Active users at Q1 2023'),
        
        html.Div(children=
            [
                html.Div(children=
                    [
                        dag.AgGrid(
                            columnDefs= [
                                {"field": "Segment", "cellStyle": cellStyle},
                                {"field": "# COUNT"}
                            ],
                            rowData= df_rfm_totals.to_dict('records'),
                            className="ag-theme-balham",
                            columnSize="sizeToFit",
                            style={'height': '231px'},
                        )
                    ], style={'padding': 10, 'flex-basis': '230px', 'flex-grow': 0, 'flex-shrink': 0}
                ),
                html.Div(children=
                    [
                        dcc.Graph(
                            id='scatter-m-f',
                            figure = px.scatter(
                                df_rfm[df_rfm["Monetary"] < 10000], 
                                x='Frequency', 
                                y='Monetary', 
                                color='CLASS',
                                color_discrete_map=segs_map
                            ).update_traces(showlegend=False).update_layout(margin={"t": 0, "r": 0, "b": 0, "l": 0}, autosize=True),
                            config = {'displayModeBar': False}
                        )
                    ], style={'padding': 10, 'flex': 1}
                ),
                html.Div(children=
                    [
                        dcc.Graph(
                            id='scatter-r-f',
                            figure = px.bar(
                                df_rfm_monet,
                                x='CLASS',
                                y='Monetary (med.)',
                                color='CLASS',
                                color_discrete_map=segs_map
                            ).update_traces(showlegend=False).update_layout(margin={"t": 0, "r": 0, "b": 0, "l": 0}, autosize=True),
                            config = {'displayModeBar': False}
                        )
                    ], style={'padding': 10, 'flex': 1}
                )
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        ),

        dcc.Markdown('## Segment inspection'),

        html.Div(children=
            [
                html.Div(children=
                    [
                        dcc.Dropdown(
                            id='filter-segment',
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
                            id="segment-stats",
                            columnDefs= [
                                {"field": "Statistics", "width": 320},
                                {"field": "Value"}
                            ],
                            rowData= df_rfm_totals[df_rfm_totals["Segment"] == "DIAMOND"].to_dict('records'),
                            className="ag-theme-balham",
                            columnSize="sizeToFit",
                            style={'height': '362px'},
                        )
                    ], style={'padding': 10, 'flex-basis': '230px', 'flex-grow': 0, 'flex-shrink': 0}
                ),
                html.Div(children=
                    [
                        dag.AgGrid(
                            id="segment-users",
                            columnDefs= [
                                {"field": header} for header in df_rfm_table.columns
                            ],
                            #rowData= df_rfm_totals.to_dict('records'),
                            defaultColDef={"sortable": True, "filter": True},
                            className="ag-theme-balham",
                            columnSize="sizeToFit",
                            style={'height': '398px'},
                        )
                    ], style={'padding': 10, 'flex': 1}
                )
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        )
    ]
)

"""
UI CALLBACKS
"""

@callback(
    Output('segment-stats', 'rowData'),
    [Input('filter-segment', 'value')]
)
def update_table(selected_segment):
    #return [{'Statistics': 'Recency', 'Value': 71.0}, {'Statistics': 'Frequency', 'Value': 5.0}, {'Statistics': 'Monetary', 'Value': 159.875}, {'Statistics': 'Avg. products', 'Value': 2.647037084713723}]
    df_return = df_rfm_stats.reset_index().reset_index(drop=True)[["index", selected_segment]]
    return [{"Statistics": row["index"], "Value": row[selected_segment]} for index, row in df_return.iterrows()]

@callback(
    Output('segment-users', 'rowData'),
    [Input('filter-segment', 'value')]
)
def update_users_table(selected_segment):
    df_return = df_rfm_table[df_rfm_table["Segment"] == selected_segment]
    return df_return.to_dict("records")


"""

layout = html.Div(
    [
        dcc.Markdown('# RFM Analysis / Active users at 2023 Q1'),
        
        html.Div(children=
            [
                html.Div(children=
                    [
                        dag.AgGrid(
                            columnDefs= [
                                {"field": "Segment", "cellStyle": cellStyle},
                                {"field": "# COUNT"}
                            ],
                            rowData= df_rfm_totals.to_dict('records'),
                            className="ag-theme-balham",
                            columnSize="sizeToFit",
                            style={'height': '300px'},
                        )
                    ], style={'padding': 10, 'flex-basis': '230px', 'flex-grow': 0, 'flex-shrink': 0}
                ),
                html.Div(children=
                    [
                        dcc.Graph(
                            id='scatter-m-f',
                            figure = px.scatter(
                                df_rfm[df_rfm["Monetary"] < 10000], 
                                x='Frequency', 
                                y='Monetary', 
                                color='CLASS',
                                color_discrete_map=segs_map
                            ).update_traces(showlegend=False).update_layout(margin={"t": 0, "r": 0, "b": 0, "l": 0}, autosize=True),
                            config = {'displayModeBar': False}
                        )
                    ], style={'padding': 10, 'flex': 1}
                ),
                html.Div(children=
                    [
                        dcc.Graph(
                            id='scatter-r-f',
                            figure = px.scatter(
                                df_rfm, 
                                x='Recency', 
                                y='Frequency', 
                                color='CLASS',
                                color_discrete_map=segs_map,
                            ).update_traces(showlegend=False).update_layout(margin={"t": 0, "r": 0, "b": 0, "l": 0}, autosize=True),
                            config = {'displayModeBar': False}
                        )
                    ], style={'padding': 10, 'flex': 1}
                )
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        )
    ]
)
"""
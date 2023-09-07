import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt 

import dash_cytoscape as cyto
import pandas as pd
import numpy as np
import os 
import json

elements_stylesheet = [
    {
        "selector": 'node',
        "style": {
            'background-color': 'gray',
            'opacity': 0.9,
            "label": "data(label)",
        }
    },
    {
        "selector": 'edge',
        'style': {
            "mid-target-arrow-color": "#0f0",
            "mid-target-arrow-shape": "vee",
        }
    },
]

# page 2 data

                    #------------------------|---------------------------#
                    # cursor support         | selection of segments     #
                    # cursor confidence      | all                       #
                    # cursor lift            | diamond/gold/silver/bronze#
                    #------------------------|---------------------------#
                    #                        |                           #
                    # network graph          |  histogram                #
                    #                        |                           #
                    #------------------------|---------------------------#
                    # cursor #rules          |                           #
                    #------------------------|---------------------------#


dash.register_page(__name__, name='Products', order=3)

path = ".//pages//"
MBA_df_DIAMOND = pd.read_csv(path + "dataframe_MBA_product_group_confidence_support_lift_threshold_1_sort_by_confidence_DIAMOND.csv")
MBA_df_GOLD = pd.read_csv(path + "dataframe_MBA_product_group_confidence_support_lift_threshold_1_sort_by_confidence_GOLD.csv")
MBA_df_SILVER = pd.read_csv(path + "dataframe_MBA_product_group_confidence_support_lift_threshold_1_sort_by_confidence_SILVER.csv")

min_support = round(min(MBA_df_DIAMOND['support']),3)
max_support = round(max(MBA_df_DIAMOND['support']),3)
min_confidence = round(min(MBA_df_DIAMOND['confidence']),3)
max_confidence = round(max(MBA_df_DIAMOND['confidence']),3)
max_support = round(max(MBA_df_DIAMOND['support']),3)
min_lift = round(min(MBA_df_DIAMOND['lift']),1)
max_lift = round(max(MBA_df_DIAMOND['lift']),1)
MBA_df_GOLD = pd.read_csv(path + "dataframe_MBA_product_group_confidence_support_lift_threshold_1_sort_by_confidence_GOLD.csv")
min_support_GOLD = round(min(MBA_df_GOLD['support']),3)
max_support_GOLD = round(max(MBA_df_GOLD['support']),3)
min_confidence_GOLD = round(min(MBA_df_GOLD['confidence']),3)
max_confidence_GOLD = round(max(MBA_df_GOLD['confidence']),3)
max_support_GOLD = round(max(MBA_df_GOLD['support']),3)
min_lift_GOLD = round(min(MBA_df_GOLD['lift']),1)
max_lift_GOLD = round(max(MBA_df_GOLD['lift']),1)
MBA_df_SILVER = pd.read_csv(path + "dataframe_MBA_product_group_confidence_support_lift_threshold_1_sort_by_confidence_SILVER.csv")
min_support_SILVER = round(min(MBA_df_SILVER['support']),3)
max_support_SILVER = round(max(MBA_df_SILVER['support']),3)
min_confidence_SILVER = round(min(MBA_df_SILVER['confidence']),3)
max_confidence_SILVER = round(max(MBA_df_SILVER['confidence']),3)
max_support_SILVER = round(max(MBA_df_SILVER['support']),3)
min_lift_SILVER = round(min(MBA_df_SILVER['lift']),1)
max_lift_SILVER = round(max(MBA_df_SILVER['lift']),1)

def extract_item_from_df_prd_group(_prod_group):
    temp = _prod_group.replace(',frozenset({',';({')
    temp = temp.replace('frozenset({','({')
    temp = temp.replace('({','')
    temp = temp.replace('})','')
    temp = temp.replace('.0','')
    _ant = temp.split(';')[0]
    _con = temp.split(';')[1]
    return _ant,_con

def cyto_plot(N_rules=100):
    df_ = MBA_df_DIAMOND 

    print(N_rules)

    edges = []
    elements = []
    node_ids = []

    for i in range(N_rules):
        temp = extract_item_from_df_prd_group(df_.iloc[i]['product_group'])
        _a = temp[0]
        _b = temp[1]

        node_ids.append(_a)
        node_ids.append(_b)
        
        edges.append({
            'data': {
                'source': _a,
                'target': _b,
                'label': 'Miao'
            }
        })

    for n in list(set(node_ids)):
        elements.append({
            'data': {'id': n, 'label': n}
        })

    return elements + edges

"""
[
    {'data': {'id': 'one', 'label': 'Node 1'}},
    {'data': {'id': 'two', 'label': 'Node 2'}},
    {'data': {'id': 'three', 'label': 'Node 3'}},
    {'data': {'source': 'one', 'target': 'two','label': 'Node 1 to 2'}},
    {'data': {'source': 'three', 'target': 'two','label': 'Node 3 to 2'}},
]
"""

    #_weight = df_.iloc[i]['lift'] * 0.01
    #G.add_nodes_from([_a])
    #G.add_nodes_from([_b])
    #G.add_edge(_a,_b)
    #G.add_edge(_a, _b, color=colors[i] , weight = _weight)
    

def graph_gen1(N_rules=100):
    df_ = MBA_df_DIAMOND 
    #print(df_[:N_rules])
    
    G = nx.DiGraph()
    
    N = 50
    colors = np.random.rand(N_rules)

    for i in range(N_rules):
        temp = extract_item_from_df_prd_group(df_.iloc[i]['product_group'])
        _a = temp[0]
        _b = temp[1]
        _weight = df_.iloc[i]['lift'] * 0.01
        G.add_nodes_from([_a])
        G.add_nodes_from([_b])
        G.add_edge(_a,_b)
        G.add_edge(_a, _b, color=colors[i] , weight = _weight)

    #print(G.number_of_edges())
    #print(G.number_of_nodes())
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]
    _edgelist = list(G.edges())
    pos = nx.spring_layout(G, k=16, scale=1)
    
    fig = go.Layout(
        nx.draw_networkx(
            G, pos, arrows=True, edgelist=_edgelist, edge_color=colors, 
            width=weights, font_size=16, with_labels=False),
    )

    #for p in pos:  # raise text positions
    #    nx.draw_networkx_labels(G, pos, font_size=8)

    return fig

def graph_digraph(N_rules=100):
    df_ = MBA_df_DIAMOND 
    #print(df_[:N_rules])
    G = nx.DiGraph()
    for i in range(N_rules):
        #G1.add_nodes_from(["R"+str(i)])
        ##print(df_.iloc[i]['product_group'])
        temp = extract_item_from_df_prd_group(df_.iloc[i]['product_group'])
        _ant = temp[0]
        _cons = temp[1]
        G.add_nodes_from([_ant, _cons])
        G.add_edge(_ant, _cons)
    plt.show()

graph_digraph(N_rules=10)
          
def graph_gen(N_rules=100):
    df_ = MBA_df_DIAMOND 
    #print(df_[:N_rules])

    set_nodes_Nrules = []
    # #for _rules in df_[:N_rules]:
        
    for i in range(N_rules):
        #G1.add_nodes_from(["R"+str(i)])
        ##print(df_.iloc[i]['product_group'])
        temp = extract_item_from_df_prd_group(df_.iloc[i]['product_group'])
        # temp = df_.iloc[i]['product_group'].replace(',frozenset({',';({')
        # temp = temp.replace('frozenset({','({')
        # _a = temp.split(';')[0]
        # _b = temp.split(';')[1]
        set_nodes_Nrules.append(temp[0])
        set_nodes_Nrules.append(temp[1])
        # G.add_nodes_from([_a])
        # G.add_nodes_from([_b])
    set_nodes_Nrules = set(set_nodes_Nrules)
    #print(set_nodes_Nrules)
    #print(len(set_nodes_Nrules))
    G = nx.random_geometric_graph(len(set_nodes_Nrules), 0,seed=123) 
    #G.edges = dict()
    #print(G.number_of_edges())
    #print(G.number_of_nodes())
    #print(G.nodes[0])
    #print(G.nodes[1])
    #     #print(_a,_b)
    # for node in G.nodes():
    #     #print(node)
    # for a in rules.iloc[i]['antecedents']:
    #     G1.add_nodes_from([a])
    #     G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
    # for c in rules.iloc[i]['consequents']:
    #     G1.add_nodes_from([c])
    #     G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
     
    for i in range(N_rules):

        temp = extract_item_from_df_prd_group(df_.iloc[i]['product_group'])
        _ant = temp[0]
        _cons = temp[1]
        #print(_ant,_cons)
        #print(G.nodes[1+i],G.nodes[2+i])
        #G.add_edge(G.nodes[1+i],G.nodes[2+i],  weight=2)
        # temp = df_.iloc[i]['product_group'].replace(',frozenset({',';({')
        # temp = temp.replace('frozenset({','({')
        # _a = temp.split(';')[0]
        # _b = temp.split(';')[1]
        # G.add_nodes_from([_a])
        # G.add_nodes_from([_b])
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
    
    edge_trace = go.Scatter(
          x=edge_x, y=edge_y,
          line=dict(width=1, color='#888'),
          #line= dict(size=10,symbol= "arrow-bar-up", angleref="previous"),
          hoverinfo='text',
          mode='lines')
      
    node_x = []
    node_y = []
    for node in G.nodes():
          x, y = G.nodes[node]['pos']
          node_x.append(x)
          node_y.append(y)
    
    node_trace = go.Scatter(
          x=node_x, y=node_y,
          mode='markers',
          hoverinfo='text',
          marker=dict(
              showscale=True,
              # colorscale options
              #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
              #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
              #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
              colorscale='YlGnBu',
              reversescale=True,
              color=[],
              size=10,
              colorbar=dict(
                  thickness=15,
                  title='Node Connections',
                  xanchor='left',
                  titleside='right'
              ),
              line_width=2))
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    #title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

histo = px.histogram(MBA_df_DIAMOND, x="support", nbins=20)
value_rules = 10

layout = html.Div(
    [
        dbc.Row([ 
            dbc.Col(dcc.Markdown('# MBA')),
            html.Hr(),
            ],
            ),
        dbc.Row(
            [
                dbc.Col(   
                        [
                            #html.H4("Live adjustable graph-size"),
                            html.P("Support:"),
                            dcc.RangeSlider(
                                min=min_support, max=max_support, value=[min_support, max_support],
                                tooltip={"placement": "bottom", "always_visible": True},
                                id='support'),
                            html.P("Confidence:"),
                            dcc.RangeSlider(
                                min=min_confidence, max=max_confidence, value=[min_confidence, max_confidence],
                                tooltip={"placement": "bottom", "always_visible": True}, 
                                id='confidence'),
                            html.P("Lift:"),
                            dcc.RangeSlider(
                                min=min_lift, max=max_lift, value=[min_lift, max_lift],
                                tooltip={"placement": "bottom", "always_visible": True},
                                id='lift'),
                            ],
                        ),
                #dbc.Col(html.Div("One of three columns")),
                dbc.Col(
                        [ 
                            html.Div("RMF segments"),
                            dcc.Dropdown(
                                    options=["no segment","Diamond","Gold","Silver","Bronze"],
                                    value="Diamond", #if ccy else "USD",
                                    multi=False,
                                    placeholder="Select a segment",
                                    id="segment",
                                    ),
                            ],
                    ),
                html.Hr(), 

            ],
        ),
        dbc.Row(
            [ 
                dbc.Col(   
                        [
                            html.H4("Graph with rules:"),
                            cyto.Cytoscape(
                                id='network_graph',
                                elements=[],
                                layout={'name': 'grid'},
                                stylesheet=elements_stylesheet
                            ),
                            #dcc.Graph(figure=graph_gen1(value_rules), id="network_graph"),
                            html.P("First #rules:"),
                            dcc.Slider(id="first_rules_slider", min=1, max=21, value=7, step=1,
                                       marks={1: '1', 3: '3', 5: '5', 7: '7', 9: '9', 11: '11', 
                                              13: '13', 15: '15', 17: '17', 19: '19', 21: '21'})
                            ],
                ),
                 dbc.Col(   
                         [
                            html.H4("Histogram with support"),
                            dcc.Graph(figure=histo, id='hist'),
 
                        ],
                ),   
            ],
        ),
    ],
) 

@callback( # Update support value
    Output('support', 'value'),
    Input('segment','value'),
    )
def update_support_value(value):
    if value == "Silver":
        result_ = [min_support_SILVER,max_support_SILVER]
    if value == "Gold":
        result_ = [min_support_GOLD,max_support_GOLD]
    if value == "Diamond":
        result_ = [min_support, max_support]
    # temp_df = MBA_df[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_
@callback( # Update support min
    Output('support', 'min'),
    Input('segment','value'),
    )
def update_support_min(value):
    if value == "Silver":
        result_ = min_support_SILVER
    if value == "Gold":
        result_ = min_support_GOLD
    if value == "Diamond":
        result_ = min_support
    # temp_df = MBA_df[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_
@callback( # Update support max
    Output('support', 'max'),
    Input('segment','value'),
    )
def update_support_max(value):
    if value == "Silver":
        result_ = max_support_SILVER
    if value == "Gold":
        result_ = max_support_GOLD
    if value == "Diamond":
        result_ = max_support
    # temp_df = MBA_df[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_

@callback( # Update confidence value
    Output('confidence', 'value'),
    Input('segment','value'),
    )
def update_confidence_value(value):
    ##print(value)
    if value == "Silver":
        result_ = [min_confidence_SILVER,max_confidence_SILVER]
    if value == "Gold":
        result_ = [min_confidence_GOLD,max_confidence_GOLD]
    if value == "Diamond":
        result_ = [min_confidence, max_confidence]
    # temp_df = MBA_dfx[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_
@callback( # Update support min
    Output('confidence', 'min'),
    Input('segment','value'),
    )
def update_confidence_min(value):
    ##print(value)
    if value == "Silver":
        result_ = min_confidence_SILVER
    if value == "Gold":
        result_ = min_confidence_GOLD
    if value == "Diamond":
        result_ = min_confidence
    # temp_df = MBA_df[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_
@callback( # Update support max
    Output('confidence', 'max'),
    Input('segment','value'),
    )
def update_confidence_max(value):
    ##print(value)
    if value == "Silver":
        result_ = max_confidence_SILVER
    if value == "Gold":
        result_ = max_confidence_GOLD
    if value == "Diamond":
        result_ = max_confidence
    # temp_df = MBA_df[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_


@callback( # Update lift min
    Output('lift', 'min'),
    Input('segment','value'),
    )
def update_lift_min(value):
    ##print(value)
    if value == "Silver":
        result_ = min_lift_SILVER
    if value == "Gold":
        result_ = min_lift_GOLD
    if value == "Diamond":
        result_ = min_lift
    # temp_df = MBA_df[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_
@callback( # Update lift max
    Output('lift', 'max'),
    Input('segment','value'),
    )
def update_lift_max(value):
    ##print(value)
    if value == "Silver":
        result_ = max_lift_SILVER
    if value == "Gold":
        result_ = max_lift_GOLD
    if value == "Diamond":
        result_ = max_lift
    # temp_df = MBA_df[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_

@callback( # Update lift value
    Output('lift', 'value'),
    Input('segment','value'),
    )
def update_lift_value(value):
    #print(value)
    if value == "Silver":
        result_ = [min_lift_SILVER,max_lift_SILVER]
    if value == "Gold":
        result_ = [min_lift_GOLD,max_lift_GOLD]
    if value == "Diamond":
        result_ = [min_lift, max_lift]  
    # temp_df = MBA_df[MBA_df['support'] <= max_sup]
    # temp_df = temp_df[temp_df['support'] >= min_sup]
    # histo = px.histogram(temp_df, x='support', nbins=20)
    return result_

@callback( # Update high value
    Output('hist', 'figure'),
    Input('segment','value'),
    Input('support','value'),
    )
def update_hist_value(segment_value, support_value):
    if segment_value == "Silver":
        df_ = MBA_df_SILVER
    if segment_value == "Gold":
        df_ = MBA_df_GOLD
    if segment_value == "Diamond":
        df_ = MBA_df_DIAMOND  
    min_sup = support_value[0]
    max_sup = support_value[1]
    temp_df = df_[df_['support'] <= max_sup]
    temp_df = temp_df[temp_df['support'] >= min_sup]
    histo = px.histogram(temp_df, x='support', nbins=20)
    return histo

@callback( # Update graph min
    Output('network_graph', 'elements'),
    Input('first_rules_slider','value'),
    )
def update_graph_Nrules(value):
    elements = cyto_plot(value)
    """
    elements = []
    elements.append({'data': {'id': 'one', 'label': 'one'}})
    elements.append({'data': {'id': 'two', 'label': 'two'}})
    elements.append({'data': {'id': 'three', 'label': 'three'}})
    elements.append({'data': {'source': 'one', 'target': 'two','label': 'Miao'}})
    elements.append({'data': {'source': 'three', 'target': 'two','label': 'Miao'}})
    """
    return elements

    #print(elements)
    #return elements

# @callback( # Update graph min
#     Output('network_graph', 'data'),
#     Input('segment','value'),
#     )
# def update_graph(value):
#     aaa=0
#     return aaa
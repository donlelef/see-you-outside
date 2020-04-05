# Import required libraries
import pickle
from datetime import datetime
import copy
import pathlib
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html

# Multi-dropdown options
# from controls import COUNTIES, WELL_STATUSES, WELL_TYPES, WELL_COLORS
import config
import generate_data

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data_generated").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

default_config = {'displaylogo': False}


def get_slider_marks(date_start=datetime(2020, 2, 1), date_end=datetime(2020, 6, 1), freq=7):
    date_start = datetime(2020, 2, 1)
    end_date = datetime(2020, 6, 1)
    date_range = pd.date_range(date_start, end_date, freq='{}D'.format(freq))
    dict_marks = dict()
    for i, d in enumerate(date_range):
        # dict_marks[i*freq] = '{}-{}'.format(d.day,d.month)
        dict_marks[i * freq] = d.strftime('%d %b')
    return dict_marks


# Create global chart template
mapbox_access_token = "pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w"

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "See you outside",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Graphic tool for testing lockdown release strategy", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="https:devpost-laushack",  # TODO change
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Select the date to release the lockdown intervention x:",
                            className="control_label",
                        ),
                        dcc.Slider(
                            id='date-slider_0',
                            updatemode='mouseup',
                            step=1,
                            value=20,
                            min=0,
                            max=100,
                            marks=get_slider_marks(config.date_start_interventions, config.date_stop_interventions,
                                                   freq=14),
                            # tooltip={'always_visible': True},
                        ),
                        html.P(
                            "Select the date to release the lockdown intervention y:",
                            className="control_label",
                        ),
                        dcc.Slider(
                            id='date-slider_1',
                            updatemode='mouseup',
                            step=1,
                            value=0,
                            min=0,
                            max=100,
                            marks=get_slider_marks(config.date_start_interventions, config.date_stop_interventions,
                                                   freq=14),
                            # tooltip={'always_visible': True},
                        ),
                        html.P(
                            "Select the date to release the lockdown intervention z:",
                            className="control_label",
                        ),
                        dcc.Slider(
                            id='date-slider_2',
                            updatemode='mouseup',
                            step=1,
                            value=20,
                            min=0,
                            max=100,
                            marks=get_slider_marks(config.date_start_interventions, config.date_stop_interventions,
                                                   freq=14),
                            # tooltip={'always_visible': True},
                        ),
                        html.P(
                            "Select the date to release the lockdown intervention a:",
                            className="control_label",
                        ),
                        dcc.Slider(
                            id='date-slider_3',
                            updatemode='mouseup',
                            step=1,
                            value=20,
                            min=0,
                            max=100,
                            marks=get_slider_marks(config.date_start_interventions, config.date_stop_interventions,
                                                   freq=14),
                            # tooltip={'always_visible': True},
                        ),
                        # dcc.RangeSlider(
                        #     id="year_slider",
                        #     min=1960,
                        #     max=2017,
                        #     value=[1990, 2010],
                        #     className="dcc_control",
                        # ),
                        html.P("Filter by well status:", className="control_label"),
                        dcc.RadioItems(
                            id="well_status_selector",
                            options=[
                                {"label": "All ", "value": "all"},
                                {"label": "Active only ", "value": "active"},
                                {"label": "Customize ", "value": "custom"},
                            ],
                            value="active",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        )
                    ],
                    className="pretty_container four columns",
                    # className="four columns",
                    id="cross-filter-options",
                ),

                html.Div(
                    [
                        dcc.Graph(
                            id="count_graph",
                            figure=generate_data.return_plot_from_dates([20, 20, 20, 20]),
                            responsive=True,
                            config=default_config
                        )
                    ],
                    id="countGraphContainer",
                    className="pretty_container eight columns"
                ),
            ],
            className="row flex-display",
        ),

        html.Div(
            className="row flex-display",
            children=[
                html.Div([
                    html.P("Precomputed scenario:"),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No Lockdown', 'value': 'no_lockdown.csv'},
                            {'label': 'Full Lockdown', 'value': 'full_lockdown.csv'},
                            {'label': 'Plague Inc', 'value': 'plague_inc.csv'}
                        ],
                        value='no_lockdown.csv',
                        clearable=False,
                        id='scenario-selection-dropdown',
                    )
                ],
                    className="pretty_container four columns",
                    id="scenario-selection-div"
                ),
                html.Div([
                    dcc.Graph(
                        id='scenario-chart',
                        responsive=True,
                        config=default_config
                    )
                ],
                    className="pretty_container eight columns",
                    id="scenario-chart-div"
                )
            ])
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


@app.callback(
    output=dash.dependencies.Output('scenario-chart', 'figure'),
    inputs=[dash.dependencies.Input('scenario-selection-dropdown', 'value')])
def generate_scenario_plot_on_scenario_change(chosen_scenario):
    return generate_data.get_scenario_plot(chosen_scenario)


# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("count_graph", "figure")],
)


@app.callback(
    Output("count_graph", "figure"),
    [Input("date-slider_0", "value"),
     Input("date-slider_1", "value"),
     Input("date-slider_2", "value"),
     Input("date-slider_3", "value")])
def update_output(date_0, date_1, date_2, date_3):
    dates = [date_0, date_1, date_2, date_3]
    # print('Dates: {}'.format(dates))
    dates.sort()
    # print('DAtes: {}'.format(dates))
    fig = generate_data.return_plot_from_dates(dates)
    return fig


# Main
if __name__ == "__main__":
    app.run_server(debug=True)

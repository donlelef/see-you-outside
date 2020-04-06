# Import required libraries

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction, State

import generate_data

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

default_config = {'displaylogo': False}

# Create app layout
app.layout = html.Div(
    id="mainContainer",
    children=[
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(src=app.get_asset_url("LauzHack.svg"), className="responsive-img")
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "See you outside",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H6(
                                    "Optimizing lockdown release policy", style={"margin-top": "0px"}
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
                            href="https://devpost.com/software/see-you-outside",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={'margin-bottom': '25px'}
        ),
        html.Div(
            className="row flex-display",
            children=[
                html.Div([
                    html.P("How many ICUs per 1000 inhabitants?"),
                    dcc.Slider(
                        id='hospital-beds',
                        min=0.1,
                        max=10,
                        step=0.1,
                        value=1,
                        marks={v: str(v) for v in range(1, 11, 2)}
                    ),
                    html.P("How bad is lockdown - the higher, the worse?"),
                    dcc.Slider(
                        id='lockdown-penalty',
                        min=10,
                        max=1000,
                        step=10,
                        value=100,
                        marks={v: str(v) for v in range(100, 1001, 200)}
                    ),
                    html.P("How bad is filling ICUs?"),
                    dcc.Slider(
                        id='icus-penalty',
                        min=100,
                        max=10000,
                        step=10,
                        value=1000,
                        marks={v: str(v) for v in range(1000, 10001, 2000)}
                    ),
                    html.Button('Run!', id='run')],
                    className="pretty_container four columns",
                    id="commands-div"
                ),
                html.Div([
                    dcc.Loading(
                        dcc.Graph(
                            id='sir-chart',
                            config=default_config,
                            responsive=True
                        ))],
                    className="pretty_container eight columns",
                    id="sir-chart-div"
                )
            ]),
        html.Div(
            className="row flex-display",
            children=[
                html.Div([
                    dcc.Loading(
                        dcc.Graph(
                            id='hospital-chart',
                            config=default_config,
                            responsive=True
                        ))],
                    className="pretty_container one-half column",
                    id="hospital-chart-div"
                ),
                html.Div([
                    dcc.Loading(
                        dcc.Graph(
                            id='lockdown-chart',
                            config=default_config,
                            responsive=True
                        ))],
                    className="pretty_container one-half column",
                    id="lockdown-chart-div"
                )
            ])
    ])


@app.callback(
    output=[
        Output('sir-chart', 'figure'),
        Output('hospital-chart', 'figure'),
        Output('lockdown-chart', 'figure')
    ],
    state=[
        State('hospital-beds', 'value'),
        State('icus-penalty', 'value'),
        State('lockdown-penalty', 'value'),
    ],
    inputs=[Input('run', 'n_clicks')]
)
def generate_plots_on_input_change(n_clicks, hospital_beds, icus_penalty, lockdown_penalty):
    data_set = generate_data.generate_data_from_model(hospital_beds, icus_penalty, lockdown_penalty)
    plots = (
        generate_data.get_sir_plot(data_set),
        generate_data.get_hospital_plot(data_set),
        generate_data.get_lockdown_plot(data_set)
    )
    return plots


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("sir-chart", "figure"), Input("hospital-chart", "figure"), Input("lockdown-chart", "figure")],
)

# Main
if __name__ == "__main__":
    app.run_server(debug=True)

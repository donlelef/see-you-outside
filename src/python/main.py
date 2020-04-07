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
                    html.P("Percentage of initially infected people:"),
                    dcc.Slider(
                        id='I',
                        min=0,
                        max=30,
                        step=5,
                        value=5,
                        marks={v: str(v) for v in range(0, 31, 5)}
                    ),
                    html.P("Number of hospital beds (per 10'000 inhabitants):"),
                    dcc.Slider(
                        id='hospital-beds',
                        min=5,
                        max=50,
                        step=5,
                        value=5,
                        marks={v: str(v) for v in range(5, 51, 5)}
                    ),
                    html.P("How much freedom do you want (lockdown relaxation)?"),
                    dcc.Slider(
                        id='lockdown-penalty',
                        min=0,
                        max=100,
                        step=10,
                        value=10,
                        marks={v: str(v) for v in range(0, 101, 10)}
                    ),
                    html.P("How much do you want to avoid filling hospital beds?"),
                    dcc.Slider(
                        id='icus-penalty',
                        min=0,
                        max=100,
                        step=10,
                        value=10,
                        marks={v: str(v) for v in range(0, 101, 10)}
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
                            responsive=True,
                            style={'height': '360px'}
                        ))],
                    className="pretty_container eight columns",
                    id="sir-chart-div",
                    style={'height': '400px'}
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
                            responsive=True,
                            style={'height': '360px'}
                        ))],
                    className="pretty_container six columns",
                    id="hospital-chart-div",
                    style={'height': '400px'}
                ),
                html.Div([
                    dcc.Loading(
                        dcc.Graph(
                            id='lockdown-chart',
                            config=default_config,
                            responsive=True,
                            style={'height': '360px'}
                        ))],
                    className="pretty_container six columns",
                    id="lockdown-chart-div",
                    style={'height': '400px'}
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
        State('I', 'value'),
        State('hospital-beds', 'value'),
        State('icus-penalty', 'value'),
        State('lockdown-penalty', 'value'),
    ],
    inputs=[Input('run', 'n_clicks')]
)
def generate_plots_on_input_change(n_clicks, I, hospital_beds, icus_penalty, lockdown_penalty):
    data_set = generate_data.generate_data_from_model(I, hospital_beds, icus_penalty, lockdown_penalty)
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
    app.run_server(host='0.0.0.0', port=8080, debug=True)

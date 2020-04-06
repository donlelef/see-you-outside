import pathlib
from datetime import datetime


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data_generated").resolve()

date_start_interventions = datetime(2020,4,1)
date_stop_interventions = datetime(2020,6,1)

date_start_simulations = datetime(2020,3,15)

states_label_short = ['S', 'E', 'A', 'I', 'H', 'D', 'R']
states_label = ['Susceptible', 'Exposed', 'Asymptomatic', 'I', 'H', 'Dead', 'Recovered']



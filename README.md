# See You Outside

A visual tool to simulate the consequences of lockdown release policies and find the optimal policy for given requirements. 

See You Outside was developed during LauzHack 2020. More information at https://devpost.com/software/see-you-outside.

## Why?
When the most critical phase of the pandemic passes, the main issue will be deciding how to ease lockdown and allow people to get back to their activities. The case of Hong Kong suggests that the release cannot be immediate and a policy of progressive release of the lockdown is required. This scenario is unprecedented and decision-makers lack tools to quantitatively assess the consequences of policies.

## What does it do?
A partial lockdown means we can send only part of the population out at a given time, reopening progressively businesses and institutions, so that the epidemics stay under control, keeping the transmission at levels that avoid getting our hospitals overwhelmed. So how, do we do put that in place?

See you outside enables: 
1. test and visualization of the effects of different lockdown release strategies; 
2. computation of the best release strategy depending on different economic and sanitary objectives

The tool is powered by a study that builds upon existing and proven epidemic models in order to take into account the economic and sanitary consequences of different policies to release the lockdown. Very concretely, the web app allows users to interactively select different dates for re-opening schools, businesses, quarantines, etc; and observe the prediction of the evolution of the number of patients infected, in intensive care units, or deceased. 


## Getting Started
The main part of See You Outside is a [Dash](https://dash.plotly.com/) App.

### Running the app locally
You need Python 3.7 to run the app. In a Python or conda environment,
install the requirements.

Go to the root of the project and run

```
cd dash_app
pip install -r requirements.txt
```

That's it. You can run the app

```
python main.py
```

### Using the app on Google Cloud
An instance of the application is currently deployed on
Google Cloud, at https://see-you-outside.appspot.com.
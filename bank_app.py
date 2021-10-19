# Import Packages
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import math
import statistics
import numpy as np
import random
 
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


# Import Dataset

bank = pd.read_csv('bank.csv')

# Data Cleaning

bank.columns

bank = bank.rename(columns = {'y': 'deposit'})
bank['deposit'] = (bank['deposit']=='yes').astype(int)
bank['default'] = (bank['default']=='yes').astype(int)
bank['housing'] = (bank['housing']=='yes').astype(int)
bank['loan'] = (bank['loan']=='yes').astype(int)
bank['deposit'].value_counts()
bank['job'] = bank['job'].replace({'management': 'Employed', 'blue-collar': 'Employed',
                                  'technician': 'Employed', 'admin.': 'Employed',
                                  'services': 'Employed', 'retired': 'Unemployed',
                                  'self-employed': 'Self-employed', 'entrepreneur':'Employed',
                                  'unemployed': 'Unemployed', 'housemaid': 'Employed',
                                  'student': 'Unemployed', 'unknown': 'Employed'})

b_day = [1, 2, 999]
l_day = ['1', '0']
bank['day'] = pd.to_numeric(bank['day'])
bank['day'] = pd.cut(bank['day'], bins = b_day, labels = l_day, include_lowest = True)

b_age = [0, 21, 65, 99999]
l_age = ['Youth', 'Adult', 'Senior']

bank['age'] = pd.to_numeric(bank['age'])
bank['age'] = pd.cut(bank['age'], bins = b_age, labels = l_age, include_lowest = True)

bank['age'] = bank['age'].replace({'Youth': '0',
                                  'Adult': '1',
                                  'Senior': '2'})

bank['month'] = bank['month'].replace({'jan': '1','feb':'2','mar':'3','apr':'4','may':'5',
                                        'jun':'6','jul':'7','aug':'8','sep':'9','oct':'10',
                                        'nov':'11','dec':'12'
                                   })
bank['month'] = pd.to_numeric(bank['month'])

b_month = [0, 3, 6, 9, 12]
l_month = ['Q1','Q2','Q3','Q4']

bank['month'] = pd.to_numeric(bank['month'])
bank['month'] = pd.cut(bank['month'], bins = b_month, labels = l_month, include_lowest = True)

bank['month'] = bank['month'].replace({'Q1': '1',
                                       'Q2': '2',
                                       'Q3': '3',
                                       'Q4': '4'})
cat_df = bank[['job', 'marital']]
dummy = pd.get_dummies(cat_df)

bank = pd.concat([bank,dummy], axis = 1)

bank = bank.drop(['job', 'marital'], axis = 1)

# Pdays
b = [-1, 0, 99999]
l = ['0', '1']

bank['pdays'] = bank['pdays'].astype(int)
bank['pdays'] = pd.cut(bank['pdays'], bins = b, labels = l, include_lowest = True)

#Education
bank['education'] = bank['education'].replace({'primary': '0',
                                               'secondary':'0.25',
                                               'tertiary':'0.5',
                                               'unknown':'1'})
bank['education'] = bank['education'].astype(float)

#Contact and POutcome
bank['contact'] = bank['contact'].replace({'cellular': '0',
                                            'telephone':'1',
                                            'unknown':'2'
                                           })

bank['poutcome'] = bank['poutcome'].replace({'failure': '0',
                                              'other':'1',
                                              'success':'2',
                                              'unknown':'3'
                                             })
bank.dtypes


#X = bank.drop(['deposit'], 1)
#y = bank['deposit']

#X, y = datasets.make_classification(random_state = 1)
#X, test_X, y, test_y = train_test_split(bank, labels, test_size = .3, train_size = .7, random_state = 1)
#train_X,valid_X, train_y, valid_y = train_test_split(x, y, test_size = .3, train_size = .4, random_state = 1)
#train_X, valid_X, train_y, valid_y, test_X, test_y = train_test_split(X, y, train_size = .4, test_size = 0.3, random_state = 1)
# Train, Valid, Test Datasets
 
train, temp = train_test_split(bank, train_size=1700, random_state=1)
valid, test = train_test_split(temp, train_size=1488, random_state=1)



# Set Variables
 
train_X = train.drop(['deposit'], 1)
train_y = train['deposit']
 
valid_X = valid.drop(['deposit'], 1)
valid_y = valid['deposit']
 
test_X = test.drop(['deposit'], 1)
test_y = test['deposit']

print('Training   : ', train_X.shape)
print('Validation : ', valid_X.shape)
print('Test : ', test_X.shape)

# Standardize Variables
 
# train
train_X = StandardScaler().fit_transform(train_X)
 
# valid
valid_X = StandardScaler().fit_transform(valid_X)
 

# Model Algorithms
 
dTree = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 30, max_leaf_nodes = 10)
#logit_reg = LogisticRegression(penalty="l2", C = 1e42, solver = 'liblinear')
logit_reg = LogisticRegressionCV(penalty="l2", solver='lbfgs', cv = 5, max_iter = 1000)
bagging = BaggingClassifier(dTree, max_samples = 0.5, max_features = 0.5)
adaboost = AdaBoostClassifier(n_estimators = 100, base_estimator = dTree)
rf = RandomForestClassifier(n_estimators = 100, random_state = 1)
param_grid = {'hidden_layer_sizes': list(range(2, 10)),}
neuralNet1 = MLPClassifier(activation = 'logistic', solver = 'lbfgs', random_state = 1, max_iter = 5000)
gridSearch = GridSearchCV(neuralNet1, param_grid, cv = 5, n_jobs = -1)
LDA = LinearDiscriminantAnalysis()

#scaleInput = MinMaxScaler()
#scaleInput.fit(train_X * 1.0)

#neuralNet2 = MLPClassifier(hidden_layer_sizes = (10), activation = 'logistic', solver = 'lbfgs', max_iter = 5000, random_state = 1)

 
 
# Models for ROC Curve
 
MODELS = {'Logistic Regression': logit_reg,
          'Decision Tree': dTree,
          'Bagging': bagging,
          'Boosting': adaboost,
          'Random Forest': rf,
          'Neural Network': gridSearch,
          'Linear Discriminant Analysis': LDA}


#Final Results and Model Selection

final_predictiors = ['education', 'duration', 'campaign', 'poutcome']

test_X = test[final_predictiors]
test_y = test['deposit']

# Standardize the Data
test_X_norm = StandardScaler().fit_transform(test_X)

# App Development

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
app.title = "Bank Analytics: Predictive Modeling"

colors = {
        'background': '#ffffff',
        'text': '#125650'
}

app.layout = html.Div(style={'backgroundColor': colors['background']},
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Bank Analytics", className = "header-title",
                    style={
                        'textAlign': 'center',
                        'color': colors['text'],
                        'fontSize': '40px',
                        }
                ),
                html.P(
                    children="Analyze the trends of term deposit subscriptions "
                    "based on the effects of market campaigning",
                    className='header-description',
                    style={
                        'textAlign': 'center',
                        'color': colors['text'],
                        'fontSize': '28px',
                        }
                ),
            ],
            className='header-title',
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children='Model Type:', className='menu-title'),
                        dcc.Dropdown(
                            id = 'model-name',
                            options = [
                                {'label': x, 'value': x}
                                for x in MODELS
                            ],
                            value='Logistic Regression',
                            clearable = False,
                            className='dropdown',
                        ),
                    ],
                ),
            ],        
            className = 'menu',
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id = 'Train-Model', config={'displayModeBar': False},
                    ),
                    className='card',
                ),
                html.Div(
                    children=dcc.Graph(
                        id = 'Final-Model', config={'displayModeBar': False},
                    ),
                    className='card',
                ),
            ],
            className='wrapper'
        ),
    ]
)
 

@app.callback(
    [Output('Train-Model', 'figure'), Output('Final-Model', 'figure')],
    [Input('model-name', 'value')]
)

def update_charts(name):
    model1 = MODELS[name]
    model1.fit(train_X, train_y)
    
    y_score = model1.predict_proba(valid_X)[:,1]
    
    fpr, tpr, thresholds = metrics.roc_curve(valid_y, y_score)
    score = metrics.auc(fpr, tpr)
    
    fig1 = px.area(
        x = fpr, y  = tpr,
        title=f'ROC Curve - Trained Model (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate',
            y='True Positive Rate'))
    fig1.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)
    
    model2 = MODELS[name]
    model2.fit(test_X_norm, test_y)
    
    y_score = model2.predict_proba(test_X_norm)[:,1]
    
    fpr, tpr, thresholds = metrics.roc_curve(test_y, y_score)
    score = metrics.auc(fpr, tpr)
    
    fig2 = px.area(
        x = fpr, y  = tpr,
        title=f'ROC Curve - Final Model (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate',
            y='True Positive Rate'))
    fig2.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)
    
    return fig1, fig2

app.run_server(debug=True)

server = app.server

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from dash import callback,no_update
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

# Sample columns for DataTable
columns = ['Pressure', 'Vrel']
columns_density = ['Pressure(Psig)', 'Pressure(Psia)', 'Bod_Old', 'Rsd_Smoothed', 'SG_Smoothed', 'Oil_Density_Calculated']
columns_separtor = [
    'Separtor_Pressure-1', 'Separtor_Temperature-1', 'Separator_GOR-1', 'Separator_SG-1',
    'Stock_Tank_GOR', 'Stock_Tank_SG', 'Stock_Tank_Oil_Gravity', 'Bofb_Old_Lab',
    'Density_at_Old_Pb', 'Bofb_Old_Density_Corrected', 'Bofb_New', 'Rsfb_New'
]
columns_ = ['Pressure', 'Vrel','Pressure(Psia)','Y-Function','Vrel_New','Oil_Compressibility(1/Psi)']
datatable_columns = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in columns
]

datatable_columns_separtor = [{'name': 'Separator_Test', 'id': 'Separator_Test', 'type': 'text','editable': False}] + [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in columns_separtor
]
data_separator = [{'Separator_Test': f'Sep_Test-{i + 1}', **{col['id']: '' for col in datatable_columns_separtor[1:]}} for i in range(10)]
datatable_columns_density = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in columns_density
]
columns_density_Extended = ['Pressure(Psig)', 'Pressure(Psia)', 'Bod_Density_Corrected_Extended', 'Rsd_Extended', 'SG_Extended', 'Oil_Density_Calculated']
datatable_columns_density_Extended = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in columns_density_Extended
]

result_datatable_columns = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    if col != 'Oil_Compressibility(1/Psi)' else {'name': col, 'id': col, 'type': 'numeric','format': Format(precision=9, scheme=Scheme.fixed)}
    for col in ['Pressure', 'Vrel', 'Pressure(Psia)', 'Y-Function', 'Vrel_New', 'Oil_Compressibility(1/Psi)']
]


# Global variable to store the DataFrame
global_df_CME = pd.DataFrame(columns=columns)
global_df_CME_new = pd.DataFrame(columns=['Pressure', 'Vrel','Pressure(Psia)','Y-Function'])

def initial_plot(x_title, y_title, x_val, y_val, plot_title, func):
    x = x_val.values
    y = y_val.values

    # Curve fitting with the passed function
    popt, _ = curve_fit(func, x, y)
    y_pred = func(x, *popt)
    r_squared = r2_score(y, y_pred)

    # Generate the equation based on the function used
    if "poly" in func.__name__:
        # Polynomial fit (e.g., y = ax^2 + bx + c)
        coefs = popt[::-1]  # Coefficients in reverse order for poly equation
        equation = "y = " + " + ".join([f"{coef:.5f}x^{i}" for i, coef in enumerate(coefs)])
    elif "model_func_hyperbolic" in func.__name__:
        equation = f"y = {popt[0]:.5f} / x + {popt[1]:.5f}"
    elif "exp" in func.__name__:
        equation = f"y = {popt[0]:.5f} * ({popt[1]:.5f} ^ ({popt[2]:.5f} * x)) + {popt[3]:.5f}"
    elif "log" in func.__name__:
        equation = f"y = {popt[0]:.5f} * log(x) + {popt[1]:.5f}"
    elif "power" in func.__name__:
        equation = f"y = {popt[0]:.5f} * x^{popt[1]:.5f}"
    elif "modified_hyperbolic" in func.__name__:
        equation = f"y = {popt[0]:.5f} / (x + {popt[1]:.5f}) + {popt[2]:.5f} * x^{popt[3]:.5f}"
    elif "Decline_hyperbolic" in func.__name__:  # Adding support for Decline_hyperbolic
        equation = f"y = {popt[0]:.5f} / (({popt[2]:.5f} * {popt[3]:.5f} * x + {popt[1]:.5f})^(1/{popt[2]:.5f}))"

    equation += f"<br>R² = {r_squared:.5f}"


    x_new = np.random.randint(np.min(x), np.max(x) + 1, 200)
    x_new = np.sort(np.concatenate(([np.min(x), np.max(x)], x_new)))

    # Create scatter plot and fitted line
    fig = go.Figure({
        'data': [
            go.Scatter(x=x, y=y, mode='markers', marker=dict(color='blue', size=10), name='Data Points',
                       customdata=np.arange(len(x))),
            go.Scatter(x=x_new, y=func(x_new, *popt), mode='lines', line=dict(color='red'), name='Fit')
        ],
        'layout': go.Layout(
            title=plot_title,
            xaxis={'title': x_title},
            yaxis={'title': y_title},
            annotations=[{
                'x': 0.05, 'y': 0.95, 'xref': 'paper', 'yref': 'paper',
                'text': equation, 'showarrow': False, 'align': 'left',
                'font': {'size': 12}, 'bordercolor': 'black',
                'borderwidth': 1, 'bgcolor': 'white', 'opacity': 0.8
            }]
        )
    })

    return fig, popt

def fitting_plot_1(click_, figure_, disabled_indices_, func):
    if click_ and figure_['data']:
        point_index = click_['points'][0]['pointIndex']
        x = np.array(figure_['data'][0]['x'])
        y = np.array(figure_['data'][0]['y'])

        # Disable or enable points
        if point_index in disabled_indices_:
            disabled_indices_.remove(point_index)
        else:
            disabled_indices_.append(point_index)

        # Color the points (blue = enabled, red = disabled)
        colors = ['blue'] * len(x)
        for ind in disabled_indices_:
            colors[ind] = 'red'

        fig = go.Figure(figure_)
        fig.data[0].marker.color = colors

        # Exclude disabled points for fitting
        fit_data = [(x[i], y[i]) for i in range(len(x)) if i not in disabled_indices_]
        if fit_data:
            x_fit, y_fit = zip(*fit_data)

            # Fit the curve with the function and unknown number of parameters
            popt, _ = curve_fit(func, x_fit, y_fit)
            y_pred = func(np.array(x_fit), *popt)
            r_squared = r2_score(np.array(y_fit), y_pred)

            # Update plot for the fit line
            fig.data[1].y = func(np.array(fig.data[1].x), *popt)

            # Generate the equation based on the function used
            if "poly" in func.__name__:
                coefs = popt[::-1]  # Coefficients in reverse order for poly equation
                equation = "y = " + " + ".join([f"{coef:.5f}x^{i}" for i, coef in enumerate(coefs)])
            elif "model_func_hyperbolic" in func.__name__:
                equation = f"y = {popt[0]:.5f} / x + {popt[1]:.5f}"
            elif "exp" in func.__name__:
                equation = f"y = {popt[0]:.5f} * ({popt[1]:.5f} ^ ({popt[2]:.5f} * x)) + {popt[3]:.5f}"
            elif "log" in func.__name__:
                equation = f"y = {popt[0]:.5f} * log(x) + {popt[1]:.5f}"
            elif "power" in func.__name__:
                equation = f"y = {popt[0]:.5f} * x^{popt[1]:.5f}"
            elif "modified_hyperbolic" in func.__name__:
                equation = f"y = {popt[0]:.5f} / (x + {popt[1]:.5f}) + {popt[2]:.5f} * x^{popt[3]:.5f}"
            elif "Decline_hyperbolic" in func.__name__:  # Adding support for Decline_hyperbolic
                equation = f"y = {popt[0]:.5f} / (({popt[2]:.5f} * {popt[3]:.5f} * x + {popt[1]:.5f})^(1/{popt[2]:.5f}))"

            equation += f"<br>R² = {r_squared:.5f}"

            # Add the annotation to the plot
            fig.update_layout(
                annotations=[{
                    'x': 0.05, 'y': 0.95, 'xref': 'paper', 'yref': 'paper',
                    'text': equation, 'showarrow': False, 'align': 'left',
                    'font': {'size': 12}, 'bordercolor': 'black',
                    'borderwidth': 1, 'bgcolor': 'white', 'opacity': 0.8
                }]
            )

    return fig, disabled_indices_, popt


def initial_plot_1(x_title, y_title, x_val, y_val, plot_title, degree):
    # Reshaping data
    x = x_val.values.reshape(-1, 1)
    y = y_val.values.reshape(-1, 1)

    # Generate new x values for predictions
    x_new = np.linspace(np.min(x), np.max(x), 200).reshape(-1, 1)

    # Polynomial transformation
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x)
    x_new_mapped = poly.transform(x_new)

    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    x_new_mapped_scaled = scaler_poly.transform(x_new_mapped)

    # Create and train the model
    model = Ridge(alpha=0)  # Ridge regression model
    model.fit(X_train_mapped_scaled, y)

    # Predictions and metrics
    y_pred_train = model.predict(X_train_mapped_scaled)  # Predictions on the training set
    y_pred_new = model.predict(x_new_mapped_scaled)      # Predictions on new x values
    train_mse = mean_squared_error(y, y_pred_train) / 2  # Mean squared error
    r_squared = r2_score(y, y_pred_train)                # R² value

    # Get model coefficients and intercept
    intercept = model.intercept_[0]
    coefficients = model.coef_.flatten()

    # Combine intercept and coefficients into a single array or dictionary
    fit_params = {
        'intercept': intercept,
        'coefficients': coefficients,
        'r_squared': r_squared
    }

    # Construct equation string
    equation_terms = [f"{round(coeff, 5)} * x^{i+1}" for i, coeff in enumerate(coefficients)]
    equation = "y = " + f"{round(intercept, 5)} + " + " + ".join(equation_terms)

    # Plot the data and the regression curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.flatten(), y=y.flatten(), mode='markers', marker=dict(color='blue', size=10), name='Data Points'))
    fig.add_trace(go.Scatter(x=x_new.flatten(), y=y_pred_new.flatten(), mode='lines', line=dict(color='red'), name='Fit'))
    
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        annotations=[
            dict(x=0.05, y=0.95, xref='paper', yref='paper', text=f"{equation}<br>R² = {round(r_squared, 5)}", 
                 showarrow=False, align='left', font=dict(size=12), bordercolor='black', borderwidth=1, 
                 bgcolor='white', opacity=0.8)
        ]
    )

    # Return the plot and fit parameters
    return fig, fit_params,X_train_mapped_scaled.tolist(), x_new_mapped_scaled.tolist(),model,poly,scaler_poly

def fitting_plot(click_, figure_, disabled_indices_, X_poly_mapped, x_new_mapped):
    if click_ and figure_['data']:
        point_index = click_['points'][0]['pointIndex']
        x = np.array(figure_['data'][0]['x'])
        y = np.array(figure_['data'][0]['y'])

        # Disable or enable points
        if point_index in disabled_indices_:
            disabled_indices_.remove(point_index)
        else:
            disabled_indices_.append(point_index)

        # Color the points (blue = enabled, red = disabled)
        colors = ['blue'] * len(x)
        for ind in disabled_indices_:
            colors[ind] = 'red'

        fig = go.Figure(figure_)
        fig.data[0].marker.color = colors

        # Exclude disabled points for fitting
        fit_data = [(X_poly_mapped[i], y[i]) for i in range(len(x)) if i not in disabled_indices_]

        if fit_data:
            X_fit, y_fit = zip(*fit_data)

            # Refitting the model using the pre-computed polynomial features
            model_ = Ridge(alpha=0)  # Ridge regression model
            model_.fit(X_fit, y_fit)
            y_hat_new = model_.predict(x_new_mapped)  # Predict new y values based on updated model

            # Update the plot with the new fit line
            fig.data[1].y = y_hat_new.flatten()  # Update the y-values of the fit line

            # Get the updated equation and R²
            intercept = model_.intercept_
            coefficients = model_.coef_.flatten()
            r_squared = model_.score(X_fit, y_fit)
            fit_params = {
            'intercept': intercept,
            'coefficients': coefficients,
            'r_squared': r_squared
                                    }

            # Construct updated equation
            equation_terms = [f"{round(coeff, 5)} * x^{i+1}" for i, coeff in enumerate(coefficients)]
            equation = "y = " + f"{round(intercept, 5)} + " + " + ".join(equation_terms)

            # Add the annotation with the updated equation and R²
            fig.update_layout(
                annotations=[{
                    'x': 0.05, 'y': 0.95, 'xref': 'paper', 'yref': 'paper',
                    'text': f'{equation}<br>R²: {r_squared:.5f}', 'showarrow': False, 'align': 'left',
                    'font': {'size': 12}, 'bordercolor': 'black', 'borderwidth': 1,
                    'bgcolor': 'white', 'opacity': 0.8
                }]
            )

    return fig, disabled_indices_,fit_params,model_

# Sample column headers for DataTable
columns_Bod_1 = ['Pressure(Psig)', 'VrelD', 'Bod', 'Rsd', 'SG', 'Z-Factor', 'Eg_Lab', 'Oil_Density(gm/cc)', 'Oil_Viscosity(cp)']
columns_1 = ['Pressure(Psig)', 'VrelD', 'Bod', 'Rsd', 'SG', 'Z-Factor', 'Eg_Lab', 'Oil_Density', 'Oil_Viscosity(cp)',
            'Pressure(Psia)']
columns_Bod = ['Pressure(Psig)','Pressure(Psia)', 'VrelD', 'Bod', 'Vrel_Smoothed','Bod_New']
datatable_columns_Bod_1 = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in columns_Bod_1
]

datatable_columns_Bod = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in columns_Bod
]

datatable_columns_Z = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Z-Factor','Eg_Lab','Z-Factor_Smoothed','Eg','Gas Formation Volume Factor']
]
datatable_columns_Rsd = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Rsd','Rsd_Smoothed']
]
datatable_columns_SG = [
    {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
    for col in ['Pressure(Psig)', 'Pressure(Psia)', 'SG','SG_Corrected','SG_Smoothed','SG_Cum_Smoothed']
]
global_df_DL = pd.DataFrame(columns=columns_1)


# Define your model function
def model_func(x, a, b):
    return a * x + b
# Main tabs layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("PVT QualiCheck", style={
            'font-family': 'Arial, sans-serif',
            'font-size': '36px',
            'text-align': 'center',
            'margin-top': '20px',
            'font-weight': 'bold',
            'color': '#007BFF'
        }), width=12)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Constant Mass Expansion',
                        children=html.Div([html.Br(),dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    # Title above the DataTable
                                    html.P("Paste your data here:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                    # DataTable
                                    dash_table.DataTable(
                                        id='output-table',
                                        columns=datatable_columns,
                                        data=[{col: '' for col in columns} for _ in range(15)],  # Start with empty rows for pasting
                                        persistence=True,
                                        persistence_type='session',
                                        persisted_props=['data'],
                                        editable=True,
                                        row_deletable=True,
                                        style_table={'width': '100%', 'overflowX': 'auto','height': '400px', 'paddingLeft': '10px'},
                                        # Added padding
                                        style_cell={
                                            'fontFamily': 'Arial',  # Set font to Arial
                                            'fontSize': '14px',  # Adjust font size
                                            'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                            'height': '20px', 'padding': '2px', 'lineHeight': '20px'
                                        },
                                        style_data={
                                            'whiteSpace': 'normal', 'height': '20px',
                                        }
                                    ),
                                    # Inputs, Submit, and Clear buttons below the DataTable
                                    html.Div([
                                        html.Div([
                                            html.Label("Old Pb:", style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            dcc.Input(id='old-pb-input', type='number'),
                                        ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                        html.Div([
                                            html.Label("New Pb:", style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            dcc.Input(id='new-pb-input', type='number'),
                                        ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                        html.Div([
                                            html.Button('Submit', id='submit-button', n_clicks=0,
                                                        className='btn btn-primary'),
                                            html.Button('Clear', id='clear-button', n_clicks=0,
                                                        className='btn btn-secondary', style={'margin-left': '10px'})
                                        ], style={'display': 'inline-block'})
                                    ], style={'margin-top': '10px'}),
                                ], width=2),  # Content in 2-column width

                            dbc.Col(
                                [
                                    dcc.Graph(
                                        id='scatter-plot',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title=dict(text='V_Relative vs. Pressure Below Pb',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'V_Relative'},)
                                        },style={'height': '600px'}
                                    )
                                ],
                                width=5,

                            ),
                                dbc.Col([
                            dcc.Graph(
                                id='scatter-plot_1',
                                figure={
                                    'data': [],
                                    'layout': go.Layout(title=dict(text='V_Relative vs. Pressure Above Pb',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'V_Relative'},)
                                },
                                style={'height': '600px'}
                            ),
                                ], width=5),
                            ]),
                            dcc.Store(id='disabled-indices', data=[]),  # Store to keep track of disabled points
                            dcc.Store(id='disabled-indices_1', data=[]),
                            dcc.Store(id='cme-new-store', data=[]),
                            dcc.Store(id='slope_b', data=[]),
                            dcc.Store(id='intercept_b', data=[]),
                            dcc.Store(id='slope_a', data=[]),
                            dcc.Store(id='intercept_a', data=[]),
                            dcc.Store(id='store-data_Vrel', storage_type='memory'),
                            # Add space between tabs and the next content
                            html.Br(),

                            # Third row with the result DataTable
                            dbc.Row([
                                dbc.Col(width=3),  # Empty column for spacing
                                    html.P("Result and Summary of CME:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                dbc.Col(dash_table.DataTable(
                                    id='result-table',
                                    columns=result_datatable_columns,
                                    data=[],  # Start with no data
                                    editable=True,
                                    style_table={'width': '100%', 'overflowX': 'auto', 'paddingLeft': '10px'},
                                    # Added padding
                                    style_cell={
                                        'fontFamily': 'Arial',  # Set font to Arial
                                        'fontSize': '14px',  # Adjust font size
                                        'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                    }
                                ), width=6),
                                dbc.Col(width=3),  # Empty column for spacing
                            ]),
                        ], fluid=True)
                        ]), style={
                        'font-family': 'Arial', 'font-size': '14px'
                    }),
                dbc.Tab(label='Differntial Liberation', children=[
                    dbc.Tabs([
                        dbc.Tab(label='Oil Formation Volume Factor',
                        children=html.Div([html.Br(),dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    # Title above the DataTable
                                    html.P("Paste your data here:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                    # DataTable
                                    dash_table.DataTable(
                                        id='output-table_d',
                                        columns=datatable_columns_Bod_1,
                                        data=[{col: '' for col in columns_Bod_1} for _ in range(15)],  # Start with empty rows
                                        persistence=True,
                                        persistence_type='session',
                                        persisted_props=['data'],
                                        editable=True,
                                        row_deletable=True,
                                        style_table={'width': '100%', 'overflowX': 'auto','height': '400px', 'paddingLeft': '10px'},
                                        # Added padding
                                        style_cell={
                                            'fontFamily': 'Arial',  # Set font to Arial
                                            'fontSize': '14px',  # Adjust font size
                                            'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                            'height': '20px', 'padding': '2px', 'lineHeight': '20px'
                                        },
                                        style_data={
                                            'whiteSpace': 'normal', 'height': '20px',
                                        }
                                    ),
                                    # Inputs, Submit, and Clear buttons below the DataTable
                                    html.Div([
                                        html.Div([
                                            html.Label("Old Pb:", style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            dcc.Input(id='old-pb-input_d', type='number'),
                                        ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                        html.Div([
                                            html.Label("New Pb:", style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            dcc.Input(id='new-pb-input_d', type='number'),
                                        ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                        html.Div([
                                            html.Label("Reservoir_Temperature:", style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            dcc.Input(id='Temperature_d', type='number'),
                                        ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                        html.Br(),
                                        html.Div([
                                            html.Button('Submit', id='submit-button_d', n_clicks=0,
                                                        className='btn btn-primary'),
                                            html.Button('Clear', id='clear-button_d', n_clicks=0,
                                                        className='btn btn-secondary', style={'margin-left': '10px'})
                                        ], style={'display': 'inline-block'})
                                    ], style={'margin-top': '10px'}),
                                ], width=6),  # Content in 2-column width

                            dbc.Col(
                                [
                                            html.Div([
                                            html.Button('Show_Plots', id='submit-button_Bod', n_clicks=0,
                                                        className='btn btn-primary'),
                                            html.Button('Clear_Plots', id='clear-button_Bod', n_clicks=0,
                                                        className='btn btn-secondary', style={'margin-left': '10px'})
                                        ], style={'display': 'inline-block'}),
                                    html.P("Select the curvefit:",
                                   style={"textDecoration": "underline"}),
                                dcc.RadioItems(id='Bod-checklist', value='Logarithmic',
                                options=[{'label':x, 'value':x}
                                   for x in ['Logarithmic','Polynomial']],
                                            inline =True,style={'display': 'flex', 'gap': '20px'},labelClassName="mr-3"),
                                dcc.RadioItems(id='Bod-subchecklist', value='2',
                                options=[{'label':x, 'value':x}
                                   for x in ['1','2','3','4','5']],
                                            inline =True,style={'display': 'flex', 'gap': '15px', 'margin-left': '100px'},labelClassName="mr-3"),
                                    dcc.Graph(
                                        id='scatter-plot_Bod',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title=dict(text='Logdelta_P vs. Log V_relative',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'log_deltaP'},
                                        yaxis={'title':'log_V_Relative'},)
                                        },style={'height': '600px'}
                                    )
                                ],
                                width=6,

                            ),
                            ]),
                            dcc.Store(id='disabled-indices_Bod', data=[]),
                            dcc.Store(id='popt_Bod', data=[]),
                            dcc.Store(id='global_df_DL_store'),
                            dcc.Store(id='Bod_Density', data=[]),
                            dcc.Store(id='x_mapped_Bod', data=[]),
                            dcc.Store(id='x_new_mapped_Bod', data=[]),
                            dcc.Store(id='Bod_smooth_pressure_mapped_scaled', data=[]),
                            dcc.Store(id='Bod_smooth_pressure_mapped_scaled_Density', data=[]),
                            # Add space between tabs and the next content
                            html.Br(),

                            # Third row with the result DataTable
                            dbc.Row([
                                    html.P("Result and Summary of Oil Formation Volume Factor:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                dbc.Col(dash_table.DataTable(
                                    id='result-table_Bod',
                                    columns=datatable_columns_Bod,
                                    data=[],  # Start with no data
                                    editable=True,
                                    style_table={'width': '100%', 'overflowX': 'auto', 'paddingLeft': '10px'},
                                    # Added padding
                                    style_cell={
                                        'fontFamily': 'Arial',  # Set font to Arial
                                        'fontSize': '14px',  # Adjust font size
                                        'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                    }
                                ), width=6),

                                dbc.Col(
                                [
                                    dcc.Graph(
                                        id='Bod_Comparison',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title=dict(text='Oil Formation Volume Factor Comparison Plot',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'Oil Formation Volume Factor'},)
                                        },style={'height': '600px'}
                                    )
                                ],
                                width=6,

                            ),

                            ]),
                        ], fluid=True)
                        ]), style={
                        'font-family': 'Arial', 'font-size': '14px'
                    }),
                        dbc.Tab(label='Solution GOR',
                        children=html.Div([html.Br(),dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                            html.Div([
                                            html.Button('Show_Plots', id='submit-button_Rsd', n_clicks=0,
                                                        className='btn btn-primary'),
                                            html.Button('Clear_Plots', id='clear-button_Rsd', n_clicks=0,
                                                        className='btn btn-secondary', style={'margin-left': '10px'})
                                        ], style={'display': 'inline-block'}),
                                    # Title above the DataTable
                                    html.P("Solution GOR Data:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                    # DataTable
                                    dash_table.DataTable(
                                        id='output-table_Rsd',
                                        columns=[{'name': col, 'id': col} for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Rsd']],
                                        data=[],
                                        persistence=True,
                                        persistence_type='session',
                                        persisted_props=['data'],
                                        editable=False,  # Make the table non-editable
                                        row_deletable=False,  # Rows are not deletable by the user
                                        style_table={'width': '100%', 'overflowX': 'auto','height': '400px', 'paddingLeft': '10px'},
                                        # Added padding
                                        style_cell={
                                            'fontFamily': 'Arial',  # Set font to Arial
                                            'fontSize': '14px',  # Adjust font size
                                            'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                            'height': '20px', 'padding': '2px', 'lineHeight': '20px'
                                        },
                                        style_data={
                                            'whiteSpace': 'normal', 'height': '20px',
                                        }
                                    ),
                                ], width=2),  # Content in 2-column width
                            dbc.Col(
                                [
                                    dcc.Graph(
                                        id='Rsd_30_90',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title=dict(text='Solution_GOR_30-90',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'Solution GOR'},)
                                        },style={'height': '600px'}
                                    )
                                ],
                                width=5,

                            ),

                            dbc.Col(
                                [
                                            html.Div([
                                            html.Button('Show_Plots', id='submit-button_Rsd_1', n_clicks=0,
                                                        className='btn btn-primary'),
                                            html.Button('Clear_Plots', id='clear-button_Rsd_1', n_clicks=0,
                                                        className='btn btn-secondary', style={'margin-left': '10px'})
                                        ], style={'display': 'inline-block'}),
                                html.P("Select the curvefit:",
                                   style={"textDecoration": "underline"}),
                                dcc.Checklist(id='Rsd-checklist', value=['Hyperbolic'],
                                options=[{'label':x, 'value':x}
                                   for x in ['Hyperbolic','Polynomial']],
                                            inline =True,style={'display': 'flex', 'gap': '20px'},labelClassName="mr-3"),
                                dcc.RadioItems(id='Rsd-subchecklist', value='2',
                                options=[{'label':x, 'value':x}
                                   for x in ['1','2','3','4','5']],
                                            inline =True,style={'display': 'flex', 'gap': '15px', 'margin-left': '100px'},labelClassName="mr-3"),

                                    dcc.Graph(
                                        id='fig_Rsd',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title=dict(text='Solution GOR vs. Pressure',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'Solution GOR'},)
                                        },style={'height': '600px'}
                                    )
                                ],
                                width=5,

                            ),
                            ]),
                            dcc.Store(id='disable_indices_Rsd', data=[]),
                            dcc.Store(id='popt_Rsd', data=[]),
                            dcc.Store(id='disable_indices_Rsd_1', data=[]),
                            dcc.Store(id='popt_Rsd_1', data=[]),
                            dcc.Store(id='scaler_poly_Rsd', data=[]),
                            dcc.Store(id='x_mapped_Rsd', data=[]),
                            dcc.Store(id='x_new_mapped_Rsd', data=[]),
                            dcc.Store(id='x_new_Rsd', data=[]),
                            dcc.Store(id='x_new_Rsd_mapped_scaled', data=[]),
                            dcc.Store(id='Rsdb_new_S', data=[]),
                            # Add space between tabs and the next content
                            html.Br(),
                            dbc.Row([
                                    html.P("Result and Summary of Solution GOR:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                dbc.Col(dash_table.DataTable(
                                    id='result-table_Rsd',
                                    columns=datatable_columns_Rsd,
                                    data=[],  # Start with no data
                                    editable=True,
                                    style_table={'width': '100%', 'overflowX': 'auto', 'paddingLeft': '10px'},
                                    # Added padding
                                    style_cell={
                                        'fontFamily': 'Arial',  # Set font to Arial
                                        'fontSize': '14px',  # Adjust font size
                                        'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                    }
                                ), width=6),

                                dbc.Col(
                                [
                                    dcc.Graph(
                                        id='Rsd_Final',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title=dict(text='Smoothed Solution GOR vs. Pressure',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'Solution GOR'},)
                                        },style={'height': '600px'}
                                    )
                                ],
                                width=6,

                            ),

                            ]),
                        ], fluid=True)
                        ]), style={
                        'font-family': 'Arial', 'font-size': '14px'
                    }),
                        dbc.Tab(label='Z-Factor',
                        children=html.Div([html.Br(),dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                            html.Div([
                                            html.Button('Show_Plots', id='submit-button_Z', n_clicks=0,
                                                        className='btn btn-primary'),
                                            html.Button('Clear_Plots', id='clear-button_Z', n_clicks=0,
                                                        className='btn btn-secondary', style={'margin-left': '10px'})
                                        ], style={'display': 'inline-block'}),
                                    # Title above the DataTable
                                    html.P("Z-Factor data:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                    # DataTable
                                    dash_table.DataTable(
                                        id='output-table_Z',
                                        columns=[{'name': col, 'id': col} for col in
                                                 ['Pressure(Psig)', 'Pressure(Psia)', 'Z-Factor','Eg_Lab']],
                                        data=[],
                                        persistence=True,
                                        persistence_type='session',
                                        persisted_props=['data'],
                                        editable=False,  # Make the table non-editable
                                        row_deletable=False,  # Rows are not deletable by the user
                                        style_table={'width': '100%', 'overflowX': 'auto','height': '400px', 'paddingLeft': '10px'},
                                        # Added padding
                                        style_cell={
                                            'fontFamily': 'Arial',  # Set font to Arial
                                            'fontSize': '14px',  # Adjust font size
                                            'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                            'height': '20px', 'padding': '2px', 'lineHeight': '20px'
                                        },
                                        style_data={
                                            'whiteSpace': 'normal', 'height': '20px',
                                        }
                                    ),
                                ], width=6),  # Content in 2-column width

                            dbc.Col(
                                [
                                html.P("Select the curvefit:",
                                   style={"textDecoration": "underline"}),
                                dcc.Checklist(id='Z-checklist', value=['Hyperbolic'],
                                options=[{'label':x, 'value':x}
                                   for x in ['Hyperbolic','Polynomial']],
                                            inline =True,style={'display': 'flex', 'gap': '20px'},labelClassName="mr-3"),
                                dcc.RadioItems(id='Z-subchecklist', value='2',
                                options=[{'label':x, 'value':x}
                                   for x in ['1','2','3','4','5']],
                                            inline =True,style={'display': 'flex', 'gap': '15px', 'margin-left': '100px'},labelClassName="mr-3"),

                                    dcc.Graph(
                                        id='fig_Z',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title=dict(text='Z-Factor vs. Pressure',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'Z-Factor'},)
                                        },style={'height': '600px'}
                                    )
                                ],
                                width=6,

                            ),
                            ]),
                            dcc.Store(id='disable_indices_Z', data=[]),
                            dcc.Store(id='popt_Z', data=[]),
                            dcc.Store(id='scaler_poly_Z', data=[]),
                            dcc.Store(id='x_mapped_Z', data=[]),
                            dcc.Store(id='x_new_mapped_Z', data=[]),
                            dcc.Store(id='x_new_Z', data=[]),
                            dcc.Store(id='x_new_Z_mapped_scaled', data=[]),
                            # Add space between tabs and the next content
                            html.Br(),
                            dbc.Row([
                                    html.P("Result and Summary of  Z-Factor:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                dbc.Col(dash_table.DataTable(
                                    id='result-table_Z',
                                    columns=datatable_columns_Z,
                                    data=[],  # Start with no data
                                    editable=True,
                                    style_table={'width': '100%', 'overflowX': 'auto', 'paddingLeft': '10px'},
                                    # Added padding
                                    style_cell={
                                        'fontFamily': 'Arial',  # Set font to Arial
                                        'fontSize': '14px',  # Adjust font size
                                        'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                    }
                                ), width=6),

                                dbc.Col(
                                [
                                    dcc.Graph(
                                        id='Z_Eg',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title=dict(text='Smoothed and Extended Z-factor vs. Pressure',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'Z-factor, Gas Formation Volume Factor, Eg'},)
                                        },style={'height': '600px'}
                                    )
                                ],
                                width=6,

                            ),

                            ]),
                        ], fluid=True)
                        ]), style={
                        'font-family': 'Arial', 'font-size': '14px'
                    }),
                        dbc.Tab(label='Specific Gravity',
                                children=html.Div([html.Br(), dbc.Container([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Button('Show_Plots', id='submit-button_SG', n_clicks=0,
                                                            className='btn btn-primary'),
                                                html.Button('Clear_Plots', id='clear-button_SG', n_clicks=0,
                                                            className='btn btn-secondary',
                                                            style={'margin-left': '10px'})
                                            ], style={'display': 'inline-block'}),
                                            # Title above the DataTable
                                            html.P("Specific Gravity Data:", style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            # DataTable
                                            dash_table.DataTable(
                                                id='output-table_SG',
                                                columns=[{'name': col, 'id': col} for col in
                                                         ['Pressure(Psig)', 'Pressure(Psia)', 'SG','Rsd']],
                                                data=[],
                                                persistence=True,
                                                persistence_type='session',
                                                persisted_props=['data'],
                                                editable=False,  # Make the table non-editable
                                                row_deletable=False,  # Rows are not deletable by the user
                                                style_table={'width': '100%', 'overflowX': 'auto', 'height': '400px',
                                                             'paddingLeft': '10px'},
                                                # Added padding
                                                style_cell={
                                                    'fontFamily': 'Arial',  # Set font to Arial
                                                    'fontSize': '14px',  # Adjust font size
                                                    'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                                    'height': '20px', 'padding': '2px', 'lineHeight': '20px'
                                                },
                                                style_data={
                                                    'whiteSpace': 'normal', 'height': '20px',
                                                }
                                            ),
                                        ], width=2),  # Content in 2-column width
                                        dbc.Col(
                                            [
                                                html.P("Select the curvefit:",
                                                       style={"textDecoration": "underline"}),
                                                dcc.Checklist(id='SG-checklist_1', value=['Hyperbolic'],
                                                              options=[{'label': x, 'value': x}
                                                                       for x in ['Hyperbolic', 'Polynomial']],
                                                              inline=True, style={'display': 'flex', 'gap': '20px'},
                                                              labelClassName="mr-3"),
                                                dcc.RadioItems(id='SG-subchecklist_1', value='2',
                                                               options=[{'label': x, 'value': x}
                                                                        for x in ['1', '2', '3', '4', '5']],
                                                               inline=True,
                                                               style={'display': 'flex', 'gap': '15px',
                                                                      'margin-left': '100px'},
                                                               labelClassName="mr-3"),
                                                dcc.Graph(
                                                    id='Rsd_SG',
                                                    figure={
                                                        'data': [],
                                                        'layout': go.Layout(title=dict(text='Solution_GOR vs Pressure',
                                                                                       font=dict(family="Arial Black",
                                                                                                 size=13,
                                                                                                 color="black")),
                                                                            xaxis={'title': 'Pressure'},
                                                                            yaxis={'title': 'Solution GOR'}, )
                                                    }, style={'height': '600px'}
                                                )
                                            ],
                                            width=5,

                                        ),

                                        dbc.Col(
                                            [
                                                html.Div([
                                                    html.Button('Show_Plots', id='submit-button_SG_1', n_clicks=0,
                                                                className='btn btn-primary'),
                                                    html.Button('Clear_Plots', id='clear-button_SG_1', n_clicks=0,
                                                                className='btn btn-secondary',
                                                                style={'margin-left': '10px'})
                                                ], style={'display': 'inline-block'}),
                                                html.P("Select the curvefit:",
                                                       style={"textDecoration": "underline"}),
                                                dcc.Checklist(id='SG-checklist', value=['Hyperbolic'],
                                                              options=[{'label': x, 'value': x}
                                                                       for x in ['H(A)', 'Poly','Exp','Log','H(B)','H(C)','Power']],
                                                              inline=True, style={'display': 'flex', 'gap': '20px'},
                                                              labelClassName="mr-3"),
                                                dcc.RadioItems(id='SG-subchecklist', value='2',
                                                               options=[{'label': x, 'value': x}
                                                                        for x in ['1', '2', '3', '4', '5']],
                                                               inline=True,
                                                               style={'display': 'flex', 'gap': '15px',
                                                                      'margin-left': '100px'},
                                                               labelClassName="mr-3"),

                                                dcc.Graph(
                                                    id='fig_SG',
                                                    figure={
                                                        'data': [],
                                                        'layout': go.Layout(
                                                            title=dict(text='Specific Gravity vs. Pressure',
                                                                       font=dict(family="Arial Black", size=13,
                                                                                 color="black")),
                                                            xaxis={'title': 'Pressure'},
                                                            yaxis={'title': 'Specific Gravity'}, )
                                                    }, style={'height': '600px'}
                                                )
                                            ],
                                            width=5,

                                        ),
                                    ]),
                                    dcc.Store(id='disable_indices_Rsd_SG', data=[]),
                                    dcc.Store(id='popt_Rsd_SG', data=[]),
                                    dcc.Store(id='SG_df_Rsd_SG', data=[]),
                                    dcc.Store(id='disable_indices_SG', data=[]),
                                    dcc.Store(id='popt_SG', data=[]),
                                    dcc.Store(id='scaler_poly_SG', data=[]),
                                    dcc.Store(id='x_mapped_SG', data=[]),
                                    dcc.Store(id='x_new_mapped_SG', data=[]),
                                    dcc.Store(id='x_mapped_SG_1', data=[]),
                                    dcc.Store(id='x_new_mapped_SG_1', data=[]),
                                    dcc.Store(id='x_new_SG', data=[]),
                                    dcc.Store(id='x_new_SG_mapped_scaled', data=[]),
                                    dcc.Store(id='Rsd_SG_smooth_pressure_mapped_scaled', data=[]),
                                    dcc.Store(id='df_7', data=[]),
                                    dcc.Store(id='Rsd_SG_Density', data=[]),
                                    dcc.Store(id='Rsdb_old_S', data=[]),
                                    
                                    # Add space between tabs and the next content
                                    html.Br(),
                                    dbc.Row([
                                        html.P("Result and Summary of Specific Gravity:", style={
                                            'font-family': 'Arial', 'font-size': '14px'
                                        }),
                                        dbc.Col(dash_table.DataTable(
                                            id='result-table_SG',
                                            columns=datatable_columns_SG,
                                            data=[],  # Start with no data
                                            editable=True,
                                            style_table={'width': '100%', 'overflowX': 'auto', 'paddingLeft': '10px'},
                                            # Added padding
                                            style_cell={
                                                'fontFamily': 'Arial',  # Set font to Arial
                                                'fontSize': '14px',  # Adjust font size
                                                'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                            }
                                        ), width=6),

                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    id='SG_Final',
                                                    figure={
                                                        'data': [],
                                                        'layout': go.Layout(
                                                            title=dict(text='Specific Gravity vs. Pressure',
                                                                       font=dict(family="Arial Black", size=13,
                                                                                 color="black")),
                                                            xaxis={'title': 'Pressure'},
                                                            yaxis={'title': 'Specific Gravity'}, )
                                                    }, style={'height': '600px'}
                                                )
                                            ],
                                            width=6,

                                        ),

                                    ]),
                                ], fluid=True)
                                                   ]), style={
                                'font-family': 'Arial', 'font-size': '14px'
                            }),

                            dbc.Tab(
                                    label='Oil Density',
                                    children=html.Div([
                                        html.Br(),
                                        dbc.Container([
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        html.Button('From Previous Analysis', id='Prev', n_clicks=0, className='btn btn-primary'),
                                                        html.Button('Fresh Data', id='Fresh', n_clicks=0, className='btn btn-secondary', style={'margin-left': '10px'})
                                                    ], style={'display': 'inline-block'}),
                                                    # DataTable
                                                    dash_table.DataTable(
                                                        id='output-table_Density',
                                                        columns=datatable_columns_density,  # Example columns
                                                        data=[{'Pressure(Psig)': '', 'Pressure(Psia)': '', 'Bod_Old': '', 'Rsd_Smoothed': '', 'SG_Smoothed': '', 'Oil_Density_Calculated': ''} for _ in range(12)],  # Empty rows by default
                                                        persistence=True,
                                                        persistence_type='session',
                                                        persisted_props=['data'],
                                                        editable=True,
                                                        row_deletable=True,
                                                        style_table={'width': '100%', 'overflowX': 'auto', 'height': '400px', 'paddingLeft': '10px'},
                                                        style_cell={'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '100px', 'maxWidth': '200px', 'height': '20px', 'padding': '2px', 'lineHeight': '20px'},
                                                        style_data={'whiteSpace': 'normal', 'height': '20px'}
                                                    ),
                                                    # Inputs, Submit, and Clear buttons below the DataTable
                                                    html.Div([
                                                        html.Div([
                                                            html.Label("SG of Stock Tank Oil After Differntial Liberation:", style={'font-family': 'Arial', 'font-size': '14px'}),
                                                            dcc.Input(id='SG', type='number'),
                                                        ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                                        html.Br(),
                                                        html.Div([
                                                            html.Button('☑️', id='submit-button_Density', n_clicks=0, className='btn btn-primary'),
                                                            html.Button('❌', id='clear-button_Density', n_clicks=0, className='btn btn-secondary', style={'margin-left': '10px'})
                                                        ], style={'display': 'inline-block'})
                                                    ], style={'margin-top': '10px'})
                                                ], width=6),  # Content in 2-column width

                                                dbc.Col([
                                                    html.P("Enter the Laboratory Measured Density Data:", style={'font-family': 'Arial', 'font-size': '14px'}),
                                                    dash_table.DataTable(
                                                        id='output-measured_Density',
                                                        columns=[{'name': col, 'id': col} for col in ['Pressure(Psig)', 'Measured_Density']],  # Example columns
                                                        data=[{'Pressure(Psig)': '', 'Measured_Density': ''} for _ in range(12)],  # Empty rows by default
                                                        persistence=True,
                                                        persistence_type='session',
                                                        persisted_props=['data'],
                                                        editable=True,
                                                        row_deletable=True,
                                                        style_table={'width': '100%', 'overflowX': 'auto', 'height': '400px', 'paddingLeft': '10px'},
                                                        style_cell={'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '100px', 'maxWidth': '200px', 'height': '20px', 'padding': '2px', 'lineHeight': '20px'},
                                                        style_data={'whiteSpace': 'normal', 'height': '20px'}
                                                    ),
                                                    html.Div([
                                                        html.Button('☑️', id='submit-button_Density_measured', n_clicks=0, className='btn btn-primary'),
                                                        html.Button('❌', id='clear-button_Density_measured', n_clicks=0, className='btn btn-secondary', style={'margin-left': '10px'})
                                                    ], style={'display': 'inline-block'})
                                                ], width=2),

                                                dcc.Store(id='Bodb_old_S', data=[]),
                                                dcc.Store(id='Bodb_new_S', data=[]),
                                                dcc.Store(id='Density_S', data=[]),

                                                dbc.Col([
                                                    html.P("Density Corrected Oil Formation Volume Factor:", style={'font-family': 'Arial', 'font-size': '14px'}),
                                                    dcc.Slider(
                                                        id='factor-slider',
                                                        min=0,
                                                        max=2,
                                                        step=0.001,
                                                        value=1,
                                                        marks={i: str(i) for i in range(3)}
                                                    ),
                                                    dcc.Input(
                                                        id='factor-input',
                                                        type='number',
                                                        min=0,
                                                        max=2,
                                                        step=0.001,
                                                        value=1,
                                                        style={'marginLeft': '20px'}
                                                    ),
                                                    dash_table.DataTable(
                                                        id='output-corrected_Density',
                                                        columns=[{'name': col, 'id': col} for col in ['Pressure(Psig)', 'Corrected_Density', 'Corrected_Bod']],  # Example columns
                                                        data=[],  # Empty rows by default
                                                        persistence=True,
                                                        persistence_type='session',
                                                        persisted_props=['data'],
                                                        editable=True,
                                                        row_deletable=True,
                                                        style_table={'width': '100%', 'overflowX': 'auto', 'height': '400px', 'paddingLeft': '10px'},
                                                        style_cell={'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '100px', 'maxWidth': '200px', 'height': '20px', 'padding': '2px', 'lineHeight': '20px'},
                                                        style_data={'whiteSpace': 'normal', 'height': '20px'}
                                                    ),
                                                ], width=4),
                                            ]),

                                            html.Br(),

                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Graph(
                                                        id='Density_corrected',
                                                        figure={
                                                            'data': [],
                                                            'layout': go.Layout(
                                                                title=dict(text='Measured vs. Calculated Density Plot', font=dict(family="Arial Black", size=13, color="black")),
                                                                xaxis={'title': 'Pressure'},
                                                                yaxis={'title': 'Oil Density'}
                                                            )
                                                        }, style={'height': '600px'}
                                                    )
                                                ], width=6),

                                                dbc.Col([
                                                    dcc.Graph(
                                                        id='Density_Corrected_Bod',
                                                        figure={
                                                            'data': [],
                                                            'layout': go.Layout(
                                                                title=dict(text='Density Corrected Oil Formation Volume Factor', font=dict(family="Arial Black", size=13, color="black")),
                                                                xaxis={'title': 'Pressure'},
                                                                yaxis={'title': 'Oil Formation Volume Factor'}
                                                            )
                                                        }, style={'height': '600px'}
                                                    )
                                                ], width=6),
                                            ]),

                                            dbc.Row([
                                                dbc.Col([    
                                                     html.Div([
                                                        html.Button('☑️', id='submit-button_Extended', n_clicks=0, className='btn btn-primary'),
                                                        html.Button('❌', id='clear-button_Extended', n_clicks=0, className='btn btn-secondary', style={'margin-left': '10px'})
                                                    ], style={'display': 'inline-block'}),
                                                    html.Br(),
                                                    dash_table.DataTable(
                                                        id='output-table_Extended_Density',
                                                        columns=datatable_columns_density_Extended,  # Example columns
                                                        data=[{'Pressure(Psig)': '', 'Pressure(Psia)': '', 'Bod_Density_Corrected_Extended': '', 'Rsd_Extended': '', 'SG_Extended': '', 'Oil_Density_Calculated': ''} for _ in range(12)],  # Empty rows by default
                                                        persistence=True,
                                                        persistence_type='session',
                                                        persisted_props=['data'],
                                                        editable=True,
                                                        row_deletable=True,
                                                        style_table={'width': '100%', 'overflowX': 'auto', 'height': '400px', 'paddingLeft': '10px'},
                                                        style_cell={'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '100px', 'maxWidth': '200px', 'height': '20px', 'padding': '2px', 'lineHeight': '20px'},
                                                        style_data={'whiteSpace': 'normal', 'height': '20px'}
                                                    ),                                                  
                                                ], style={'margin-top': '10px'}, width=6),

                                                dbc.Col([
                                                    dcc.Graph(
                                                        id='Extended_Density',
                                                        figure={
                                                            'data': [],
                                                            'layout': go.Layout(
                                                                title=dict(text='Calculated Density at New Bubblepoint Pressure', font=dict(family="Arial Black", size=13, color="black")),
                                                                xaxis={'title': 'Pressure'},
                                                                yaxis={'title': 'Oil Density'}
                                                            )
                                                        }, style={'height': '600px'}
                                                    )
                                                ], width=6),
                                            ]),
                                        ], fluid=True)
                                    ]), style={'font-family': 'Arial', 'font-size': '14px'}
                                )
                    ])
                ]),
               dbc.Tab(label='Separator Corrections & Result Summary',
                        children=html.Div([html.Br(),dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                            html.Label("Enter the Number of Separator Stages:",style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            dcc.Input(id='extra-set-input', type='number', min=1, step=1, value=1),
                                            html.Button('Update Table', id='update-button', n_clicks=0),
                                        ],style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '25px'}),
                                    
                                    html.Div([
                                            html.Div([
                                                html.Label("Bodb_Old:", style={'font-family': 'Arial', 'font-size': '14px'}),
                                                dcc.Input(id='Bodb_old-input_separator', type='number')
                                            ], style={'display': 'inline-block', 'margin-right': '10px'}),

                                            html.Div([
                                                html.Label("Bodb_New:", style={'font-family': 'Arial', 'font-size': '14px'}),
                                                dcc.Input(id='Bodb_new-input_separator', type='number')
                                            ], style={'display': 'inline-block', 'margin-right': '10px'}),

                                            html.Div([
                                                html.Label("Rsdb_Old:", style={'font-family': 'Arial', 'font-size': '14px'}),
                                                dcc.Input(id='Rsdb_old-input_separator', type='number')
                                            ], style={'display': 'inline-block', 'margin-right': '10px'}),

                                            html.Div([
                                                html.Label("Rsdb_New:", style={'font-family': 'Arial', 'font-size': '14px'}),
                                                dcc.Input(id='Rsdb_new-input_separator', type='number')
                                            ], style={'display': 'inline-block', 'margin-right': '10px'}),

                                            html.Div([
                                                html.Label("Oil_Density_at_Old_Pb (gm/cc):", style={'font-family': 'Arial', 'font-size': '14px'}),
                                                dcc.Input(id='density-input_separator', type='number')
                                            ], style={'display': 'inline-block', 'margin-right': '10px'})
                                        ], style={'margin-bottom': '15px'}),

                                        html.Div([
                                                dcc.Checklist(
                                                    options=[{'label': 'Use values from previous analysis', 'value': 'use_previous'}],
                                                    id='use-previous-checkbox',
                                                    value=[]
                                                )
                                            ], style={'margin-top': '25px'}),
                                    # Title above the DataTable
                                    html.P("Paste your data here:", style={
                                        'font-family': 'Arial', 'font-size': '14px'
                                    }),
                                    # DataTable
                                    dash_table.DataTable(
                                        id='output-table_separator',
                                        columns=datatable_columns_separtor,
                                        data=data_separator,  # Start with empty rows for pasting
                                        persistence=True,
                                        persistence_type='session',
                                        persisted_props=['data'],
                                        editable=True,
                                        row_deletable=True,
                                        style_table={'width': '100%', 'overflowX': 'auto','height': '400px', 'paddingLeft': '10px'},
                                        # Added padding
                                        style_cell={
                                            'fontFamily': 'Arial',  # Set font to Arial
                                            'fontSize': '14px',  # Adjust font size
                                            'minWidth': '50px', 'width': '100px', 'maxWidth': '200px',
                                            'height': '20px', 'padding': '2px', 'lineHeight': '20px'
                                        },
                                        style_data={
                                            'whiteSpace': 'normal', 'height': '20px',
                                        }
                                    ),
                                    # Inputs, Submit, and Clear buttons below the DataTable
                                    html.Div([
                                        html.Div([
                                            html.Label("Old Pb_separator:", style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            dcc.Input(id='old-pb-input_separtor', type='number'),
                                        ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                        html.Div([
                                            html.Label("New Pb_separator:", style={
                                                'font-family': 'Arial', 'font-size': '14px'
                                            }),
                                            dcc.Input(id='new-pb-input_separator', type='number'),
                                        ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                        html.Div([
                                            html.Button('Submit', id='submit-button_separator', n_clicks=0,
                                                        className='btn btn-primary'),
                                            html.Button('Clear', id='clear-button_separator', n_clicks=0,
                                                        className='btn btn-secondary', style={'margin-left': '10px'})
                                        ], style={'display': 'inline-block'})
                                    ], style={'margin-top': '10px'}),
                                ], width=12),  # Content in 2-column width

                            ]),], fluid=True)]), style={'font-family': 'Arial', 'font-size': '14px'}),
            ])
        ], width=12),  # Tab content covers 10 columns
          # Empty column for spacing
    ])
], fluid=True)


@callback(
    Output('output-table', 'data'),
    Output('scatter-plot', 'figure'),
    Output('scatter-plot_1', 'figure'),
    Output('disabled-indices', 'data'),
    Output('disabled-indices_1', 'data'),
    Output('slope_b', 'data'),  # Output for slope
    Output('intercept_b', 'data'),  # Output for intercept
    Output('slope_a', 'data'),  # Output for slope
    Output('intercept_a', 'data'),  # Output for intercept
    Output('cme-new-store', 'data'),
    Input('submit-button', 'n_clicks'),
    Input('clear-button', 'n_clicks'),
    Input('scatter-plot', 'clickData'),
    Input('scatter-plot_1', 'clickData'),
    State('output-table', 'data'),
    State('scatter-plot', 'figure'),
    State('scatter-plot_1', 'figure'),
    State('disabled-indices', 'data'),
    State('disabled-indices_1', 'data'),
    State('old-pb-input', 'value'),  # Add state for old Pb input
    State('new-pb-input', 'value'),  # Add state for new Pb input
    State('slope_b', 'data'),  # Output for slope
    State('intercept_b', 'data'),  # Output for intercept
    State('slope_a', 'data'),  # Output for slope
    State('intercept_a', 'data'),  # Output for intercept
    prevent_initial_call=True  # Ensure initial callback does not trigger
)
def update_data(submit_clicks, clear_clicks, clickData, clickdata_1, current_data, figure, figure_1, disabled_indices,
                disabled_indices_1, old_Pb, New_Pb, slope_b, intercept_b, slope_a, intercept_a):
    global global_df_CME
    global global_df_CME_new

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    slope, intercept = None, None  # Initialize slope and intercept

    if trigger_id == 'submit-button':

        df = pd.DataFrame(current_data)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

        # Remove any non-numeric characters (except decimal point and negative sign)
        df = df.applymap(
            lambda x: ''.join(filter(lambda ch: ch.isdigit() or ch in ['.', '-'], x)) if isinstance(x, str) else x)

        # Convert columns to float, handling errors by coercing invalid values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        global_df_CME = df  # Save DataFrame to global variable

        # Recalculate based on new data
        CME = global_df_CME.copy()
        CME['Pressure(Psia)'] = CME['Pressure'] + 14.7

        CME_b_Pb = CME[CME['Pressure'] <= old_Pb].copy()
        CME_b_Pb['Y-Function'] = (CME_b_Pb['Pressure'] - old_Pb) / (
                (CME_b_Pb['Pressure(Psia)']) * (1 - CME_b_Pb['Vrel']))
        CME_b_Pb_c = CME_b_Pb.dropna()
        CME_a_Pb = CME[CME['Pressure'] > old_Pb].copy()
        CME_a_Pb['Y-Function'] = np.nan
        CME_NewPb = pd.DataFrame(
            [{'Pressure': New_Pb, 'Vrel': np.nan, 'Pressure(Psia)': New_Pb + 14.7, 'Y-Function': np.nan}])
        CME_new = pd.concat([CME_b_Pb, CME_a_Pb, CME_NewPb], ignore_index=True)
        CME_new = CME_new.sort_values(by='Pressure', ascending=False)
        global_df_CME_new = CME_new
        print(CME_new)
        x = CME_b_Pb_c['Pressure'].values
        y = CME_b_Pb_c['Y-Function'].values
        x_a = CME_a_Pb['Pressure(Psia)'].values
        y_a = CME_a_Pb['Vrel'].values

        # Initial fit
        popt, _ = curve_fit(model_func, x, y)
        y_pred = model_func(x, *popt)
        r_squared = r2_score(y, y_pred)
        equation = f"y = {popt[0]:.5f}x + {popt[1]:.5f}<br>R² = {r_squared:.5f}"
        x_new = np.random.randint(np.min(x), np.max(x) + 1, 200)
        x_new = np.sort(np.concatenate(([np.min(x), np.max(x)], x_new)))
        # Update scatter plot
        fig = {
            'data': [
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(color='blue', size=10),  # Increase marker size
                    name='Data Points',
                    customdata=np.arange(len(x)),  # Custom data for indexing
                ),
                go.Scatter(
                    x=x_new,
                    y=model_func(x_new, *popt),
                    mode='lines',
                    line=dict(color='red'),
                    name='Fit'
                )
            ],
            'layout': go.Layout(title=dict(text='V_Relative vs. Pressure Below Pb',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'V_Relative'},

                                        annotations=[
                                            {
                                                'x': 0.05,
                                                'y': 0.95,
                                                'xref': 'paper',
                                                'yref': 'paper',
                                                'text': equation,
                                                'showarrow': False,
                                                'align': 'left',
                                                'font': {
                                                    'size': 12
                                                },
                                                'bordercolor': 'black',
                                                'borderwidth': 1,
                                                'bgcolor': 'white',
                                                'opacity': 0.8
                                            }
                                        ]
                                    )
                                }

        slope, intercept = popt  # Store the slope and intercept values

        if not CME_a_Pb.empty:
            popt_, _ = curve_fit(model_func, x_a, y_a)
            y_pred_ = model_func(x_a, *popt_)
            r_squared_ = r2_score(y_a, y_pred_)
            equation = f"y = {popt_[0]:.5f}x + {popt_[1]:.5f}<br>R² = {r_squared_:.5f}"
            x_new_ = np.random.randint(np.min(x_a), np.max(x_a) + 1, 200)
            x_new_ = np.sort(np.concatenate(([np.min(x_a), np.max(x_a)], x_new_)))
            # Update scatter plot
            fig_ = {
                'data': [
                    go.Scatter(
                        x=x_a,
                        y=y_a,
                        mode='markers',
                        marker=dict(color='blue', size=10),  # Increase marker size
                        name='Data Points',
                        customdata=np.arange(len(x_a)),  # Custom data for indexing
                    ),
                    go.Scatter(
                        x=x_new_,
                        y=model_func(x_new_, *popt_),
                        mode='lines',
                        line=dict(color='red'),
                        name='Fit'
                    )
                ],
                'layout': go.Layout(title=dict(text='V_Relative vs. Pressure Above Pb',font=dict(family="Arial Black",size=13,color="black")),
                                        xaxis={'title':'Pressure'},
                                        yaxis={'title':'V_Relative'},
                    annotations=[
                        {
                            'x': 0.95,
                            'y': 0.95,
                            'xref': 'paper',
                            'yref': 'paper',
                            'text': equation,
                            'showarrow': False,
                            'align': 'right',
                            'font': {
                                'size': 12
                            },
                            'bordercolor': 'black',
                            'borderwidth': 1,
                            'bgcolor': 'white',
                            'opacity': 0.8
                        }
                    ]
                )
            }

            slope_, intercept_ = popt_  # Store the slope and intercept values


        else:
            fig_ = go.Figure()
            slope_, intercept_ = np.nan, np.nan
            store_data = store_data or {}
            store_data.update({
                'table_data': current_data,
                'input1': old_Pb,
                'input2': New_Pb,
                'plot_figure_1': fig,
                'plot_figure_2': fig_,
                'disable_ini': [],
                'disable_ini_1': [],
                'slope_a': slope_,
                'intercept_a': intercept_,
                'slope_b': slope,
                'intercept_b': intercept,
                'Cme_df': global_df_CME_new.to_dict('records')
            })

        return df.to_dict(
            'records'), fig, fig_, [], [], slope, intercept, slope_, intercept_, global_df_CME_new.to_dict('records')



    elif trigger_id == 'clear-button':
        global_df_CME = pd.DataFrame(columns=['Pressure', 'Vrel'])  # Clear the global DataFrame
        fig = {
        'data': [],
        'layout': go.Layout(
        title='V_Relative vs. Pressure Below Pb',
        xaxis={'title': 'Pressure'},
        yaxis={'title': 'V_Relative'}
        )
        }
        fig_ = {
        'data': [],
        'layout': go.Layout(
        title='V_Relative vs. Pressure Above Pb',
        xaxis={'title': 'Pressure'},
        yaxis={'title': 'V_Relative'}
        )
        }  # Clear the plot
        return [], fig, fig_, [], [], [], [], [], [], []

    elif trigger_id == 'scatter-plot':
        if clickData and figure['data']:
            point_index = clickData['points'][0]['pointIndex']
            x = np.array(figure['data'][0]['x'])
            y = np.array(figure['data'][0]['y'])

            if point_index in disabled_indices:
                disabled_indices.remove(point_index)  # Enable the point if already disabled
            else:
                disabled_indices.append(point_index)  # Disable the point if not already disabled

            # Update the colors of the points
            colors = ['blue'] * len(x)
            for ind in disabled_indices:
                colors[ind] = 'red'

            # Convert figure dictionary back to go.Figure
            fig = go.Figure(figure)
            fig_ = go.Figure(figure_1)

            # Update the scatter plot with new colors
            fig.data[0].marker.color = colors

            # Update the fit line
            fit_data = [(x[i], y[i]) for i in range(len(x)) if i not in disabled_indices]
            if fit_data:
                x_fit, y_fit = zip(*fit_data)
                popt, _ = curve_fit(model_func, x_fit, y_fit)
                y_pred = model_func(np.array(x_fit), *popt)
                r_squared = r2_score(np.array(y_fit), y_pred)
                fig.data[1].y = model_func(np.array(fig.data[1].x), *popt)

                # Update the equation and R² value annotation
                equation = f"y = {popt[0]:.5f}x + {popt[1]:.5f}<br>R² = {r_squared:.5f}"
                fig.update_layout(annotations=[
                    {
                        'x': 0.05,
                        'y': 0.95,
                        'xref': 'paper',
                        'yref': 'paper',
                        'text': equation,
                        'showarrow': False,
                        'align': 'left',
                        'font': {
                            'size': 12
                        },
                        'bordercolor': 'black',
                        'borderwidth': 1,
                        'bgcolor': 'white',
                        'opacity': 0.8
                    }
                ])

                slope, intercept = popt  # Store the slope and intercept values
                slope_, intercept_ = slope_a, intercept_a

        return current_data, fig, fig_, disabled_indices, disabled_indices_1, slope, intercept, slope_, intercept_, global_df_CME_new.to_dict(
            'records')

    elif trigger_id == 'scatter-plot_1':
        if clickdata_1 and figure_1['data']:
            point_index = clickdata_1['points'][0]['pointIndex']
            x_a = np.array(figure_1['data'][0]['x'])
            y_a = np.array(figure_1['data'][0]['y'])

            if point_index in disabled_indices_1:
                disabled_indices_1.remove(point_index)  # Enable the point if already disabled
            else:
                disabled_indices_1.append(point_index)  # Disable the point if not already disabled

            # Update the colors of the points
            colors = ['blue'] * len(x_a)
            for ind in disabled_indices_1:
                colors[ind] = 'red'

            # Convert figure dictionary back to go.Figure
            fig_ = go.Figure(figure_1)
            fig = go.Figure(figure)

            # Update the scatter plot with new colors
            fig_.data[0].marker.color = colors

            # Update the fit line
            fit_data = [(x_a[i], y_a[i]) for i in range(len(x_a)) if i not in disabled_indices_1]
            if fit_data:
                x_fit, y_fit = zip(*fit_data)
                popt, _ = curve_fit(model_func, x_fit, y_fit)
                y_pred = model_func(np.array(x_fit), *popt)
                r_squared = r2_score(np.array(y_fit), y_pred)
                fig_.data[1].y = model_func(np.array(fig_.data[1].x), *popt)

                # Update the equation and R² value annotation
                equation = f"y = {popt[0]:.5f}x + {popt[1]:.5f}<br>R² = {r_squared:.5f}"
                fig_.update_layout(annotations=[
                    {
                        'x': 0.95,
                        'y': 0.95,
                        'xref': 'paper',
                        'yref': 'paper',
                        'text': equation,
                        'showarrow': False,
                        'align': 'right',
                        'font': {
                            'size': 12
                        },
                        'bordercolor': 'black',
                        'borderwidth': 1,
                        'bgcolor': 'white',
                        'opacity': 0.8
                    }
                ])

                slope_, intercept_ = popt  # Store the slope and intercept values
                slope, intercept = slope_b, intercept_b

        return current_data, fig, fig_, disabled_indices, disabled_indices_1, slope, intercept, slope_, intercept_, global_df_CME_new.to_dict(
            'records')

    # Run the app
    return current_data, figure, figure_1, disabled_indices, disabled_indices_1, slope_b, intercept_b, slope_a, intercept_a, [], current_data


@callback(
    Output('result-table', 'data'),
    Input('cme-new-store', 'data'),  # Add state for new Pb input
    Input('slope_b', 'data'),  # Output for slope
    Input('intercept_b', 'data'),  # Output for intercept
    Input('slope_a', 'data'),  # Output for slope
    Input('intercept_a', 'data'),  # Output for intercept
    State('old-pb-input', 'value'),  # Add state for old Pb input
    State('new-pb-input', 'value'),  # Add state for new Pb input
    prevent_initial_call=True  # Ensure initial callback does not trigger
)
def update_data(df, slope_b, intercept_b, slope_a, intercept_a, old_pb, new_pb):
    CME_new_df = pd.DataFrame(df)
    CME_below_Pb = CME_new_df[CME_new_df['Pressure'] < new_pb].copy()
    CME_above_Pb = CME_new_df[CME_new_df['Pressure'] > new_pb].copy()
    CME_at_Pb = CME_new_df[CME_new_df['Pressure'] == new_pb].copy()
    CME_below_Pb['Vrel_New'] = 1 + ((new_pb - CME_below_Pb['Pressure']) / (
                CME_below_Pb['Pressure(Psia)'] * (slope_b * CME_below_Pb['Pressure'] + intercept_b)))
    CME_at_Pb['Vrel_New'] = 1
    CME_above_Pb['Vrel_New'] = 1 - (new_pb - CME_above_Pb['Pressure']) * (slope_a)
    diff_column1 = CME_above_Pb['Vrel_New'].diff()
    diff_column2 = CME_above_Pb['Pressure'].diff()
    CME_above_Pb['Oil_Compressibility(1/Psi)'] = np.abs(diff_column1 / (diff_column2 * CME_above_Pb['Vrel_New']))
    CME_below_Pb['Oil_Compressibility(1/Psi)'] = np.nan
    CME_at_Pb['Oil_Compressibility(1/Psi)'] = np.nan
    CME = pd.concat([CME_above_Pb, CME_at_Pb, CME_below_Pb], ignore_index=True)
    result = CME.to_dict('records')
    return result


@callback(
    Output('store-data_Vrel', 'data'),
    Input('slope_a', 'data'),
    Input('intercept_a', 'data'),
)
def combine_and_store_data(data1, data2):
    CME_Vrel = {}

    # Directly store the float values with specific keys
    CME_Vrel['data1'] = data1
    CME_Vrel['data2'] = data2

    return CME_Vrel


#Tab_2_1_BOD

@callback(
    [Output('global_df_DL_store', 'data'),
     Output('output-table_d', 'data')],
    [Input('submit-button_d', 'n_clicks'),
     Input('clear-button_d', 'n_clicks')],
    [State('output-table_d', 'data'),
     State('old-pb-input_d', 'value'),
     State('new-pb-input_d', 'value'),
     State('Temperature_d', 'value'),
     ],
    prevent_initial_call=True  # Ensure initial callback does not trigger
)
def update_store(submit_clicks, clear_clicks, current_data_d, old_Pb_d, New_Pb_d,Temperature):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'submit-button_d':
        # Convert the data from the DataTable to a DataFrame
        df = pd.DataFrame(current_data_d)

        # Replace empty strings and strip leading/trailing whitespace
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

        # Remove any non-numeric characters (except decimal point and negative sign)
        df = df.applymap(
            lambda x: ''.join(filter(lambda ch: ch.isdigit() or ch in ['.', '-'], x)) if isinstance(x, str) else x)

        # Convert columns to float, handling errors by coercing invalid values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Calculate Pressure(Psia)
        df['Pressure(Psia)'] = df['Pressure(Psig)'] + 14.7
        global_df_DL = df

        # Store the DataFrame in dcc.Store
        return global_df_DL.to_dict('records'), global_df_DL.to_dict('records')

    elif trigger_id == 'clear-button_d':
        # Return an empty list to clear the stored data and DataTable
        return [], [{col: '' for col in columns_Bod_1}]


@callback(
    Output('disabled-indices_Bod', 'data'),
    Output('popt_Bod', 'data'),  # Output for slope
    Output('scatter-plot_Bod', 'figure'),
    Output('result-table_Bod', 'data'),
    Output('Bod_Comparison', 'figure'),
    Output('Bod_Density', 'data'),
    Output('x_mapped_Bod', 'data'),
    Output('x_new_mapped_Bod', 'data'),
    Output('Bod_smooth_pressure_mapped_scaled', 'data'),
    Output('Bod_smooth_pressure_mapped_scaled_Density', 'data'),
    Input('submit-button_Bod', 'n_clicks'),
    Input('clear-button_Bod', 'n_clicks'),
    Input('global_df_DL_store', 'data'),  # Add state for new Pb input
    Input('scatter-plot_Bod', 'clickData'),
    Input('Bod-checklist', 'value'),  # Add checklist as input
    Input('Bod-subchecklist', 'value'),
    Input('store-data_Vrel', 'data'),
    State('old-pb-input_d', 'value'),  # Add state for old Pb input
    State('new-pb-input_d', 'value'),  # Add state for new Pb input
    State('disabled-indices_Bod', 'data'),
    State('popt_Bod', 'data'),  # Output for slope
    State('scatter-plot_Bod', 'figure'),
    State('Bod_Comparison', 'figure'),
    State('Bod_Density', 'data'),
    State('x_mapped_Bod', 'data'),
    State('x_new_mapped_Bod', 'data'),
    State('result-table_Bod', 'data'),
    State('Bod_smooth_pressure_mapped_scaled', 'data'),
    State('Bod_smooth_pressure_mapped_scaled_Density', 'data'),
    prevent_initial_call=True  # Ensure initial callback does not trigger

)
def update_data(submit_clicks_Bod, clear_clicks_Bod, df, clickData_Bod,selected_option, selected_suboption,CME_Vrel, old_pb, new_pb, disabled_indices_Bod, popt, figure_Bod, figure_bod_c,Bod_Density,x_mapped, x_new_mapped,result_data,Bod_smooth_pressure_mapped_scaled,Bod_smooth_pressure_mapped_scaled_Density ):
    def poly_model_func(x, a, b):
        return a * x + b

    

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    
    if trigger_id in ['submit-button_Bod', 'Bod-checklist', 'Bod-subchecklist']:
        slope_CME = CME_Vrel.get('data1', 'No data for data1')
        intercept_CME = CME_Vrel.get('data2', 'No data for data2')
        
        Bod_new_df = pd.DataFrame(df)
        Bod_new = Bod_new_df[['Pressure(Psig)', 'Pressure(Psia)', 'VrelD', 'Bod']]
        # Model fitting functions based on checklist selection
        if 'Logarithmic' in selected_option:
            

            slope_Bod, intercept_Bod = None, None  # Initialize slope and intercept
            Bod_new_b = Bod_new[Bod_new['Pressure(Psig)'] <old_pb].copy()
            Bod_new_b['logP'] = np.log10(old_pb - Bod_new_b['Pressure(Psig)'])
            Bod_new_b['logVrel'] = np.log10(1 - Bod_new_b['VrelD'])
            fig, popt = initial_plot('log_deltaP', 'log_Vrel', Bod_new_b['logP'], Bod_new_b['logVrel'],
                                     'Log_Vrel vs. Log_P', poly_model_func)

            slope_Bod, intercept_Bod = popt[0], popt[1]
            print(slope_CME, intercept_CME, slope_Bod, intercept_Bod)
            Bod_new_d_b = Bod_new[Bod_new['Pressure(Psig)'] <= old_pb].copy()
            Bod_new_b_density = Bod_new[Bod_new['Pressure(Psig)'] < old_pb].copy()
            Bod_new_d_at_density = Bod_new[Bod_new['Pressure(Psig)'] == old_pb].copy()
            Bod_new_d_a_density = Bod_new[Bod_new['Pressure(Psig)'] > old_pb].copy()
            Bod_new_d_at = Bod_new[Bod_new['Pressure(Psig)'] == old_pb].copy()
            Bod_new_d_a = Bod_new[Bod_new['Pressure(Psig)'] > old_pb].copy()
            Bod_NewPb = pd.DataFrame(
                [{'Pressure(Psig)': new_pb, 'Pressure(Psia)': new_pb + 14.7, 'VrelD': np.nan, 'Bod': np.nan}])
            new_Bod = pd.concat([Bod_new_d_a, Bod_NewPb, Bod_new_d_b], ignore_index=True)
            new_Bod = new_Bod.sort_values(by='Pressure(Psig)', ascending=False)
            Bod_new_bd = new_Bod[new_Bod['Pressure(Psig)'] < new_pb].copy()
            Bod_new_a = new_Bod[new_Bod['Pressure(Psig)'] > new_pb].copy()
            Bod_new_at = new_Bod[new_Bod['Pressure(Psig)'] == new_pb].copy()
            Bod_new_bd['Vrel_Smoothed'] = 1 - (
                    (10 ** intercept_Bod) * ((new_pb - Bod_new_bd['Pressure(Psig)']) ** (slope_Bod)))
            print(Bod_new_bd['Vrel_Smoothed'])
            Bodb_old = Bod_new_d_at.loc[:, 'Bod']
            single_value_Bodbold = float(Bodb_old.iloc[0])
            bhavin = Bod_new_bd.copy()
            bhavin['Vrel_Smoothed_dummy'] = (single_value_Bodbold) * (
                    1 - ((10 ** intercept_Bod) * ((old_pb - bhavin['Pressure(Psig)']) ** (slope_Bod))))
            Bod_new_b_density['Bod_Old'] = (single_value_Bodbold) * (
                    1 - ((10 ** intercept_Bod) * ((old_pb - Bod_new_b_density['Pressure(Psig)']) ** (slope_Bod))))
            Bod_new_d_at_density['Bod_Old'] = single_value_Bodbold
            Bod_new_d_a_density['Bod_Old'] = (single_value_Bodbold) * (
                        1 - (old_pb - Bod_new_d_a_density['Pressure(Psig)']) * (slope_CME))
            Bod_Density = pd.concat([Bod_new_d_a_density, Bod_new_d_at_density, Bod_new_b_density], ignore_index=True)
            Bod_Density = Bod_Density[['Pressure(Psig)', 'Pressure(Psia)', 'Bod_Old']]
            print(Bod_Density)
            Bodb_new = np.nanmean(bhavin['Vrel_Smoothed_dummy'] / bhavin['Vrel_Smoothed'])
            Bod_new_at['Vrel_Smoothed'] = 1
            Bod_new_a['Vrel_Smoothed'] = 1 - (new_pb - Bod_new_a['Pressure(Psig)']) * (slope_CME)
            Bod_new_bd['Bod_New'] = Bod_new_bd['Vrel_Smoothed'] * Bodb_new
            Bod_new_at['Bod_New'] = Bodb_new
            Bod_new_a['Bod_New'] = Bodb_new * Bod_new_a['Vrel_Smoothed']
            Bod_DL = pd.concat([Bod_new_a, Bod_new_at, Bod_new_bd], ignore_index=True)
            result = Bod_DL.to_dict('records')
            cleaned_df = Bod_DL.dropna(subset=['Pressure(Psig)', 'Bod'])
            x_d = Bod_DL['Pressure(Psig)'].values
            y_d = Bod_DL['Bod_New'].values
            x_new_d = cleaned_df['Pressure(Psig)'].values
            y_new_d = cleaned_df['Bod'].values

            fig_d = {
                'data': [
                    go.Scatter(x=x_d, y=y_d, mode='lines+markers', marker=dict(color='blue', size=8), name='Bod_New',
                               customdata=np.arange(len(x_d)), ),
                    go.Scatter(x=x_new_d, y=y_new_d, mode='lines+markers', marker=dict(color='red', size=8),
                               name='Bod_Old')],
                'layout': go.Layout(title='Interactive Scatter Plot', xaxis={'title': 'Pressure'},
                                    yaxis={'title': 'Oil_Formation_Volume_fcator'})}

            return [], popt, fig, result, fig_d, Bod_Density.to_dict('records'),[],[],[],[]

        elif 'Polynomial' in selected_option:
            degree = int(selected_suboption)
            Bod_new_b = Bod_new[Bod_new['Pressure(Psig)'] <= old_pb].copy()
            x_Z = Bod_new_b['Pressure(Psia)']
            y_Z = Bod_new_b['Bod']

            # Update the plot with the selected model function
            fig, popt, x_mapped, x_new_mapped, model, poly, scaler_poly = initial_plot_1('Pressure', 'Oil Formation Volume Factor', x_Z,
                                                                                           y_Z, 'Oil Formation Volume Factor vs. Pressure',
                                                                                           degree)

            Bod_new_d_b = Bod_new[Bod_new['Pressure(Psig)'] <= old_pb].copy()
            Bod_new_b_density = Bod_new[Bod_new['Pressure(Psig)'] <= old_pb].copy()
            Bod_new_d_at_density = Bod_new[Bod_new['Pressure(Psig)'] == old_pb].copy()
            Bod_new_d_a_density = Bod_new[Bod_new['Pressure(Psig)'] > old_pb].copy()
            Bod_new_d_at = Bod_new[Bod_new['Pressure(Psig)'] == old_pb].copy()
            Bod_new_d_a = Bod_new[Bod_new['Pressure(Psig)'] > old_pb].copy()
            Bod_NewPb = pd.DataFrame(
                [{'Pressure(Psig)': new_pb, 'Pressure(Psia)': new_pb + 14.7, 'VrelD': np.nan, 'Bod': np.nan}])
            new_Bod = pd.concat([Bod_new_d_a, Bod_NewPb, Bod_new_d_b], ignore_index=True)
            new_Bod = new_Bod.sort_values(by='Pressure(Psig)', ascending=False)
            Bod_new_bd = new_Bod[new_Bod['Pressure(Psig)'] <= new_pb].copy()
            Bod_new_a = new_Bod[new_Bod['Pressure(Psig)'] > new_pb].copy()
            Bod_new_at = new_Bod[new_Bod['Pressure(Psig)'] == new_pb].copy()
            Bod_new_bd['Vrel_Smoothed'] = np.nan
            Bod_new_a['Vrel_Smoothed'] = np.nan
            Bod_smooth_pressure = Bod_new_bd['Pressure(Psia)'].values.reshape(-1, 1)
            Bod_smooth_pressure_mapped = poly.transform(Bod_smooth_pressure)
            Bod_smooth_pressure_mapped_scaled = scaler_poly.transform(Bod_smooth_pressure_mapped)
            Bod_smooth_p = model.predict(Bod_smooth_pressure_mapped_scaled)  # Predict new y values based on updated model
            Bod_smooth_p = Bod_smooth_p.flatten()
            Bod_new_bd['Bod_New'] = Bod_smooth_p
            Bodb_new = Bod_new_bd['Bod_New'].iat[0]
            Bod_new_a_a = Bod_new_a.copy()
            Bod_new_a_a['Vrel_Smooth'] = 1 - (new_pb - Bod_new_a_a['Pressure(Psig)']) * (slope_CME)
            Bod_new_a['Bod_New'] = Bodb_new * Bod_new_a_a['Vrel_Smooth']
            Bod_DL = pd.concat([Bod_new_a, Bod_new_bd], ignore_index=True)
            result = Bod_DL.to_dict('records')

            Bod_smooth_pressure_Density = Bod_new_b_density['Pressure(Psia)'].values.reshape(-1, 1)
            Bod_smooth_pressure_mapped_Density = poly.transform(Bod_smooth_pressure_Density)
            Bod_smooth_pressure_mapped_scaled_Density = scaler_poly.transform(Bod_smooth_pressure_mapped_Density)
            Bod_smooth_p_Density = model.predict(
                Bod_smooth_pressure_mapped_scaled_Density)  # Predict new y values based on updated model
            Bod_smooth_p_Density = Bod_smooth_p_Density.flatten()
            Bod_new_b_density['Bod_Old'] = Bod_smooth_p_Density
            Bodb_old = Bod_new_b_density['Bod_Old'].iat[0]
            Bod_new_d_a_density['Bod_Old'] = Bodb_old * (Bod_new_d_a_density['Pressure(Psia)'] * slope_CME + intercept_CME)
            Bod_Density = pd.concat([Bod_new_d_a_density, Bod_new_b_density], ignore_index=True)
            Bod_Density = Bod_Density[['Pressure(Psig)', 'Pressure(Psia)', 'Bod_Old']]
            cleaned_df = Bod_DL.dropna(subset=['Pressure(Psig)', 'Bod'])
            x_d = Bod_DL['Pressure(Psig)'].values
            y_d = Bod_DL['Bod_New'].values
            x_new_d = cleaned_df['Pressure(Psig)'].values
            y_new_d = cleaned_df['Bod'].values

            fig_d = {
                'data': [
                    go.Scatter(x=x_d, y=y_d, mode='lines+markers', marker=dict(color='blue', size=8), name='Bod_New',
                               customdata=np.arange(len(x_d)), ),
                    go.Scatter(x=x_new_d, y=y_new_d, mode='lines+markers', marker=dict(color='red', size=8),
                               name='Bod_Old')],
                'layout': go.Layout(title='Interactive Scatter Plot', xaxis={'title': 'Pressure'},
                                    yaxis={'title': 'Oil_Formation_Volume_fcator'})}
            return [], popt, fig, result, fig_d, Bod_Density.to_dict('records'),x_mapped, x_new_mapped,Bod_smooth_pressure_mapped_scaled,Bod_smooth_pressure_mapped_scaled_Density

    elif trigger_id == 'clear-button_Bod':
        fig = go.Figure()
        return [], [], fig, [], fig,[],[],[],[],[]

    elif trigger_id == 'scatter-plot_Bod':
        if 'Logarithmic' in selected_option:
            fig, disabled_indices_Bod, popt = fitting_plot_1(clickData_Bod, figure_Bod,disabled_indices_Bod, poly_model_func)
            print(popt)
            slope_Bod, intercept_Bod = popt[0], popt[1]
            slope_CME = CME_Vrel.get('data1', 'No data for data1')
            intercept_CME = CME_Vrel.get('data2', 'No data for data2')
            Bod_new_df = pd.DataFrame(df)
            Bod_new = Bod_new_df[['Pressure(Psig)', 'Pressure(Psia)', 'VrelD', 'Bod']]
            Bod_new_b_density = Bod_new[Bod_new['Pressure(Psig)'] < old_pb].copy()
            Bod_new_d_at_density = Bod_new[Bod_new['Pressure(Psig)'] == old_pb].copy()
            Bod_new_d_a_density = Bod_new[Bod_new['Pressure(Psig)'] > old_pb].copy()
            Bod_new_d_b = Bod_new[Bod_new['Pressure(Psig)'] <= old_pb].copy()
            Bod_new_d_at = Bod_new[Bod_new['Pressure(Psig)'] == old_pb].copy()
            Bod_new_d_a = Bod_new[Bod_new['Pressure(Psig)'] > old_pb].copy()
            Bod_NewPb = pd.DataFrame(
                [{'Pressure(Psig)': new_pb, 'Pressure(Psia)': new_pb + 14.7, 'VrelD': np.nan, 'Bod': np.nan}])
            new_Bod = pd.concat([Bod_new_d_a, Bod_NewPb, Bod_new_d_b], ignore_index=True)
            new_Bod = new_Bod.sort_values(by='Pressure(Psig)', ascending=False)
            Bod_new_bd = new_Bod[new_Bod['Pressure(Psig)'] < new_pb].copy()
            Bod_new_a = new_Bod[new_Bod['Pressure(Psig)'] > new_pb].copy()
            Bod_new_at = new_Bod[new_Bod['Pressure(Psig)'] == new_pb].copy()
            Bod_new_bd['Vrel_Smoothed'] = 1 - (
                        (10 ** intercept_Bod) * ((new_pb - Bod_new_bd['Pressure(Psig)']) ** (slope_Bod)))
            Bodb_old = Bod_new_d_at.loc[:, 'Bod']
            single_value_Bodbold = float(Bodb_old.iloc[0])
            Bod_new_b_density['Bod_Old'] = (single_value_Bodbold) * (
                    1 - ((10 ** intercept_Bod) * ((old_pb - Bod_new_b_density['Pressure(Psig)']) ** (slope_Bod))))
            Bod_new_d_at_density['Bod_Old'] = single_value_Bodbold
            Bod_new_d_a_density['Bod_Old'] = (single_value_Bodbold) * (
                        1 - (old_pb - Bod_new_d_a_density['Pressure(Psig)']) * (slope_CME))
            Bod_Density = pd.concat([Bod_new_d_a_density, Bod_new_d_at_density, Bod_new_b_density], ignore_index=True)
            Bod_Density = Bod_Density[['Pressure(Psig)', 'Pressure(Psia)', 'Bod_Old']]
            print(Bod_Density)
            bhavin = Bod_new_bd.copy()
            bhavin['Vrel_Smoothed_dummy'] = (single_value_Bodbold) * (
                        1 - ((10 ** intercept_Bod) * ((old_pb - bhavin['Pressure(Psig)']) ** (slope_Bod))))
            Bodb_new = np.nanmean(bhavin['Vrel_Smoothed_dummy'] / bhavin['Vrel_Smoothed'])
            Bod_new_at['Vrel_Smoothed'] = 1
            Bod_new_a['Vrel_Smoothed'] = 1 - (new_pb - Bod_new_a['Pressure(Psig)']) * (slope_CME)
            Bod_new_bd['Bod_New'] = Bod_new_bd['Vrel_Smoothed'] * Bodb_new
            Bod_new_at['Bod_New'] = Bodb_new
            Bod_new_a['Bod_New'] = Bodb_new * Bod_new_a['Vrel_Smoothed']
            Bod_DL = pd.concat([Bod_new_a, Bod_new_at, Bod_new_bd], ignore_index=True)
            result_ = Bod_DL.to_dict('records')
            cleaned_df = Bod_DL.dropna(subset=['Pressure(Psig)', 'Bod'])
            x_d = Bod_DL['Pressure(Psig)'].values
            y_d = Bod_DL['Bod_New'].values
            x_new_d = cleaned_df['Pressure(Psig)'].values
            y_new_d = cleaned_df['Bod'].values

            fig_d_ = {
                'data': [go.Scatter(x=x_d, y=y_d, mode='lines+markers', marker=dict(color='blue', size=8), name='Bod_New',
                                    customdata=np.arange(len(x_d)), ),
                         go.Scatter(x=x_new_d, y=y_new_d, mode='lines+markers', marker=dict(color='red', size=8),
                                    name='Bod_Old')],
                'layout': go.Layout(title='Interactive Scatter Plot', xaxis={'title': 'Pressure'},
                                    yaxis={'title': 'Oil_Formation_Volume_fcator'},
                                    )}
            return disabled_indices_Bod, popt, fig, result_, fig_d_, Bod_Density.to_dict('records'), [], [],[],[]

        if 'Polynomial' in selected_option:
            slope_CME = CME_Vrel.get('data1', 'No data for data1')
            intercept_CME = CME_Vrel.get('data2', 'No data for data2')
            X_poly_mapped = np.array(x_mapped)
            x_new_mapped_scaled = np.array(x_new_mapped)
            fig,disabled_indices_Bod, popt, model_ = fitting_plot(clickData_Bod, figure_Bod,disabled_indices_Bod, X_poly_mapped,
                                                                      x_new_mapped_scaled)
            df_r = pd.DataFrame(result_data)
            Bod_smooth_pressure_mapped_scaled_ = np.array(Bod_smooth_pressure_mapped_scaled )
            Bod_smooth_p = model_.predict(
                Bod_smooth_pressure_mapped_scaled_)  # Predict new y values based on updated model
            Bod_smooth_p = Bod_smooth_p.flatten()
            df_r_b = df_r[df_r['Pressure(Psig)']<= new_pb]
            df_r_a = df_r[df_r['Pressure(Psig)'] >new_pb]
            df_r_b['Bod_New'] = Bod_smooth_p
            Bodb_new = df_r_b['Bod_New'].iat[0]
            df_r_a_a=df_r_a.copy()
            df_r_a_a['Vrel_Smooth']=1 - (new_pb - df_r_a_a['Pressure(Psig)']) * (slope_CME)
            df_r_a['Bod_New'] = Bodb_new * (df_r_a_a['Vrel_Smooth'])
            Bod_DL = pd.concat([df_r_a, df_r_b], ignore_index=True)
            result_ = Bod_DL.to_dict('records')
            df_r_density = pd.DataFrame(Bod_Density)
            Bod_smooth_pressure_mapped_scaled_Density = np.array(Bod_smooth_pressure_mapped_scaled_Density)
            Bod_smooth_p_density = model_.predict(
                Bod_smooth_pressure_mapped_scaled_Density)  # Predict new y values based on updated model
            Bod_smooth_p_density = Bod_smooth_p_density.flatten()
            df_r_density_b = df_r_density[df_r_density['Pressure(Psig)'] <= old_pb]
            df_r_density_a = df_r_density[df_r_density['Pressure(Psig)'] > old_pb]
            df_r_density_b['Bod_Old'] = Bod_smooth_p_density
            Bodb_old = df_r_density_b['Bod_Old'].iat[0]
            df_r_density_a['Bod_Old'] = Bodb_old* (df_r_density_a['Pressure(Psia)'] * slope_CME + intercept_CME)
            Bod_Density = pd.concat([df_r_density_a, df_r_density_b], ignore_index=True)
            fig_d_ = go.Figure(figure_bod_c)
            fig_d_.data[0].x = Bod_DL['Pressure(Psig)'].values
            fig_d_.data[0].y = Bod_DL['Bod_New'].values
        return disabled_indices_Bod, popt,fig, result_, fig_d_,Bod_Density.to_dict('records'),X_poly_mapped.tolist(),x_new_mapped_scaled.tolist(),Bod_smooth_pressure_mapped_scaled_,Bod_smooth_pressure_mapped_scaled_Density
    return disabled_indices_Bod,popt,figure_Bod, [], figure_bod_c,Bod_Density,[],[],[],[]

#Z-Factor
@callback(
    Output('output-table_Z', 'data'),
    Output('disable_indices_Z', 'data'),
    Output('fig_Z', 'figure'),
    Output('popt_Z', 'data'),
    Output('scaler_poly_Z', 'data'),
    Output('x_mapped_Z', 'data'),
    Output('x_new_mapped_Z', 'data'),
    Output('result-table_Z', 'data'),
    Output('Z_Eg', 'figure'),
    Output('x_new_Z', 'data'),
    Output('x_new_Z_mapped_scaled', 'data'),
    Input('submit-button_Z', 'n_clicks'),
    Input('clear-button_Z', 'n_clicks'),
    Input('fig_Z', 'clickData'),
    Input('Z-checklist', 'value'),  # Add checklist as input
    Input('Z-subchecklist', 'value'),  # Add sub-checklist as input
    State('output-table_d', 'data'),
    State('old-pb-input_d', 'value'),
    State('new-pb-input_d', 'value'),
    State('Temperature_d', 'value'),
    State('disable_indices_Z', 'data'),
    State('fig_Z', 'figure'),
    State('popt_Z', 'data'),
    State('scaler_poly_Z', 'data'),
    State('x_mapped_Z', 'data'),
    State('x_new_mapped_Z', 'data'),
    State('output-table_Z', 'data'),
    State('result-table_Z', 'data'),
    State('Z_Eg', 'figure'),
    State('x_new_Z', 'data'),
    State('x_new_Z_mapped_scaled', 'data'),
    prevent_initial_call=True
)
def update_store(submit_clicks, clear_clicks, click_Z, selected_option, selected_suboption, current_data, old_Pb, New_Pb,Reservoir_Temperature, disable_indices_Z, fig_Z, popt,z_smooth_pressure_mapped_scaled,x_mapped,x_new_mapped, Z_data,result_data,Z_Eg,x_new_Z_,x_new_Z_mapped_scaled_):
    def model_func_hyperbolic(x, a, b, c):
        return (a / (x + b)) + c
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id in ['submit-button_Z', 'Z-checklist', 'Z-subchecklist']:
        # Model fitting functions based on checklist selection
        if 'Hyperbolic' in selected_option:
            model_func = model_func_hyperbolic
            df = pd.DataFrame(current_data)
            df = df.dropna(subset=['Pressure(Psig)', 'Z-Factor','Eg_Lab'],how = 'all')
            df['Pressure(Psia)'] = df['Pressure(Psig)'] + 14.7
            Z_df = df[['Pressure(Psig)', 'Pressure(Psia)', 'Z-Factor','Eg_Lab']]
            Z_df_Clean = Z_df[~Z_df[['Pressure(Psia)', 'Z-Factor']].isna().any(axis=1)]
            # Get values for x and y
            x_Z = Z_df_Clean['Pressure(Psia)']
            y_Z = Z_df_Clean['Z-Factor']
            print(x_Z,y_Z)
            fig_Z, popt = initial_plot('Pressure', 'Z-Factor', x_Z, y_Z, 'Z-Factor vs. Pressure', model_func)
            Z_result_df = Z_df.copy()
            new_pb_row = pd.DataFrame({'Pressure(Psig)': [New_Pb], 'Pressure(Psia)': [New_Pb+14.7],'Z-Factor':[np.nan]})
            df_result = pd.concat([Z_result_df, new_pb_row], ignore_index=True)
            df_result = df_result.drop_duplicates(subset='Pressure(Psig)', keep='first')
            df_result.sort_values(by='Pressure(Psig)', ascending=False,inplace = True)
            z_smooth_pressure = df_result['Pressure(Psia)'].values
            df_result['Z-Factor_Smoothed'] = model_func_hyperbolic(z_smooth_pressure, *popt)
            Temperature_R = (Reservoir_Temperature+273.15)*(9/5)
            df_result['Eg'] = (35.37*df_result['Pressure(Psia)'])/(df_result['Z-Factor_Smoothed']*Temperature_R)
            df_result['Gas Formation Volume Factor'] = 1/df_result['Eg']

            x_new_Z = np.random.randint(np.min(z_smooth_pressure), np.max(z_smooth_pressure) + 1, 100)
            x_new_Z = np.sort(np.concatenate(([np.min(z_smooth_pressure), np.max(z_smooth_pressure)], x_new_Z)))
            y_new_Z = model_func_hyperbolic(x_new_Z, *popt)
            Eg_y = (35.37*x_new_Z)/(y_new_Z*Temperature_R)
            Bg_y = 1/Eg_y
            x = df_result['Pressure(Psia)'][~df_result['Z-Factor'].isna()].values
            y1 = df_result['Z-Factor'][~df_result['Z-Factor'].isna()].values  # First y-axis (left)
            y2 = df_result['Eg_Lab'][~df_result['Eg_Lab'].isna()].values  # Second y-axis (right) 

            # Create the figure with two y-axes
            Z_Eg = {
                'data': [
                    go.Scatter(x=x_new_Z, y=y_new_Z, mode='lines',name='Z-Factor_Smoothed', customdata=np.arange(len(x)), ),
                    go.Scatter(x=x, y=y1, mode='markers',marker=dict(color='red', size=10),name='Z-Factor_Lab', customdata=np.arange(len(x)), ),
                    go.Scatter(x=x_new_Z, y=Bg_y, mode='lines',name='Bg', customdata=np.arange(len(x)), ),
                    go.Scatter(x=x_new_Z, y=Eg_y, mode='lines', name='Eg_Smoothed', customdata=np.arange(len(x)), yaxis='y2' ),
                    go.Scatter(x=x, y=y2, mode='markers', marker=dict(color='blue', size=10),name='Eg_Lab', customdata=np.arange(len(x)), yaxis='y2' ),
                ],
                'layout': go.Layout(
                    title='Z-Factor and Gas Formation Volume Factor Smoothed',xaxis={'title': 'Pressure (Psia)'},yaxis={'title': 'Z-Factor, Bg'},yaxis2={'title': 'Eg','overlaying': 'y', 'side': 'right'  })}
            
            
            return Z_df.to_dict('records'), [], fig_Z, popt,[],[],[],df_result.to_dict('records'),Z_Eg,[],[]
        
        elif 'Polynomial' in selected_option:
            degree = int(selected_suboption)
            # Process and clean data
            df = pd.DataFrame(current_data)
            df = df.dropna(subset=['Pressure(Psig)', 'Z-Factor','Eg_Lab'],how = 'all')
            df['Pressure(Psia)'] = df['Pressure(Psig)'] + 14.7
            Z_df = df[['Pressure(Psig)', 'Pressure(Psia)', 'Z-Factor','Eg_Lab']]
            Z_df_Clean = Z_df[~Z_df[['Pressure(Psia)', 'Z-Factor']].isna().any(axis=1)]

            # Get values for x and y
            x_Z = Z_df_Clean['Pressure(Psia)']
            y_Z = Z_df_Clean['Z-Factor']

            # Update the plot with the selected model function
            fig_Z, popt,x_mapped,x_new_mapped,model,poly,scaler_poly = initial_plot_1('Pressure', 'Z-Factor', x_Z, y_Z, 'Z-Factor vs. Pressure', degree)
            Z_result_df = Z_df.copy()
            new_pb_row = pd.DataFrame({'Pressure(Psig)': [New_Pb], 'Pressure(Psia)': [New_Pb+14.7],'Z-Factor':[np.nan]})
            df_result = pd.concat([Z_result_df, new_pb_row], ignore_index=True)
            df_result = df_result.drop_duplicates(subset='Pressure(Psig)', keep='first')
            df_result.sort_values(by='Pressure(Psig)', ascending=False,inplace = True)
            z_smooth_pressure = df_result['Pressure(Psia)'].values.reshape(-1,1)
            print(z_smooth_pressure)
            z_smooth_pressure_mapped = poly.transform(z_smooth_pressure)
            z_smooth_pressure_mapped_scaled = scaler_poly.transform(z_smooth_pressure_mapped)
            z_smooth_p = model.predict(z_smooth_pressure_mapped_scaled)  # Predict new y values based on updated model
            z_smooth_p = z_smooth_p.flatten()
            df_result['Z-Factor_Smoothed'] = z_smooth_p 
            Temperature_R = (Reservoir_Temperature+273.15)*(9/5)
            df_result['Eg'] = (35.37*df_result['Pressure(Psia)'])/(df_result['Z-Factor_Smoothed']*Temperature_R)
            df_result['Gas Formation Volume Factor'] = 1/df_result['Eg']

            x_new_Z = np.random.randint(np.min(df_result['Pressure(Psia)']), np.max(df_result['Pressure(Psia)']) + 1, 100)
            x_new_Z = np.sort(np.concatenate(([np.min(df_result['Pressure(Psia)']), np.max(df_result['Pressure(Psia)'])], x_new_Z)))
            x_new_Z_= x_new_Z.copy().flatten()
            x_new_Z = x_new_Z.reshape(-1,1)
            x_new_Z_mapped = poly.transform(x_new_Z)
            x_new_Z_mapped_scaled = scaler_poly.transform(x_new_Z_mapped)
            y_new_Z = model.predict(x_new_Z_mapped_scaled)  # Predict new y values based on updated model
            y_new_Z = y_new_Z.flatten()
            Eg_y = (35.37*x_new_Z_)/(y_new_Z*Temperature_R)
            Bg_y = 1/Eg_y
            x = df_result['Pressure(Psia)'][~df_result['Z-Factor'].isna()].values
            y1 = df_result['Z-Factor'][~df_result['Z-Factor'].isna()].values  # First y-axis (left)
            y2 = df_result['Eg_Lab'][~df_result['Eg_Lab'].isna()].values  # Second y-axis (right) 

            # Create the figure with two y-axes
            Z_Eg= {
                'data': [
                    go.Scatter(x=x_new_Z_, y=y_new_Z, mode='lines',name='Z-Factor_Smoothed', customdata=np.arange(len(x)), ),
                    go.Scatter(x=x, y=y1, mode='markers',marker=dict(color='red', size=10),name='Z-Factor_Lab', customdata=np.arange(len(x)), ),
                    go.Scatter(x=x_new_Z_, y=Bg_y, mode='lines',name='Bg', customdata=np.arange(len(x)), ),
                    go.Scatter(x=x_new_Z_, y=Eg_y, mode='lines', name='Eg_Smoothed', customdata=np.arange(len(x)), yaxis='y2' ),
                    go.Scatter(x=x, y=y2, mode='markers', marker=dict(color='blue', size=10),name='Eg_Lab', customdata=np.arange(len(x)), yaxis='y2' ),
                ],
                'layout': go.Layout(
                    title='Z-Factor and Gas Formation Volume Factor Smoothed',xaxis={'title': 'Pressure (Psia)'},yaxis={'title': 'Z-Factor, Bg'},yaxis2={'title': 'Eg','overlaying': 'y', 'side': 'right'  })}
            

            return Z_df.to_dict('records'), [], fig_Z, popt,z_smooth_pressure_mapped_scaled,x_mapped,x_new_mapped,df_result.to_dict('records'),Z_Eg,x_new_Z,x_new_Z_mapped_scaled

    elif trigger_id == 'clear-button_Z':
        return [{col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Z-Factor','Eg_Lab']}], [], go.Figure(), [],[],[],[],[{col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Z-Factor','Eg_Lab','Z-Factor_Smoothed','Eg','Gas Formation Volume Factor']}],go.Figure(),[],[]

    elif trigger_id == 'fig_Z':
        if 'Hyperbolic' in selected_option:
            fig_Z, disable_indices_Z, popt = fitting_plot_1(click_Z, fig_Z, disable_indices_Z, model_func_hyperbolic)
            df_r = pd.DataFrame(result_data)
            z_smooth_pressure = df_r['Pressure(Psia)'].values
            df_r['Z-Factor_Smoothed'] = model_func_hyperbolic(z_smooth_pressure, *popt)
            Temperature_R = (Reservoir_Temperature+273.15)*(9/5)
            df_r['Eg'] = (35.37*df_r['Pressure(Psia)'])/(df_r['Z-Factor_Smoothed']*Temperature_R)
            df_r['Gas Formation Volume Factor'] = 1/df_r['Eg']

            x_new_Z = np.random.randint(np.min(z_smooth_pressure), np.max(z_smooth_pressure) + 1, 100)
            x_new_Z = np.sort(np.concatenate(([np.min(z_smooth_pressure), np.max(z_smooth_pressure)], x_new_Z)))
            y_new_Z = model_func_hyperbolic(x_new_Z, *popt)
            Eg_y = (35.37*x_new_Z)/(y_new_Z*Temperature_R)
            Bg_y = 1/Eg_y
            Z_Eg = go.Figure(Z_Eg)
            Z_Eg.data[0].x = x_new_Z
            Z_Eg.data[0].y =y_new_Z
            
            Z_Eg.data[2].x = x_new_Z  
            Z_Eg.data[2].y = Bg_y
            
            Z_Eg.data[3].x = x_new_Z  
            Z_Eg.data[3].y = Eg_y

            return Z_data, disable_indices_Z, fig_Z, popt,[],[],[],df_r.to_dict('records'),Z_Eg,[],[]

        elif 'Polynomial' in selected_option:
            X_poly_mapped = np.array(x_mapped)
            x_new_mapped_scaled = np.array(x_new_mapped)
            fig_Z, disable_indices_Z, popt,model_= fitting_plot(click_Z, fig_Z, disable_indices_Z,X_poly_mapped, x_new_mapped_scaled)
            df_r = pd.DataFrame(result_data)
            z_smooth_pressure_mapped_scaled_=np.array(z_smooth_pressure_mapped_scaled)
            z_smooth_p = model_.predict(z_smooth_pressure_mapped_scaled_)  # Predict new y values based on updated model
            z_smooth_p = z_smooth_p.flatten()
            df_r['Z-Factor_Smoothed'] = z_smooth_p 
            Temperature_R = (Reservoir_Temperature+273.15)*(9/5)
            df_r['Eg'] = (35.37*df_r['Pressure(Psia)'])/(df_r['Z-Factor_Smoothed']*Temperature_R)
            df_r['Gas Formation Volume Factor'] = 1/df_r['Eg']
            
            x_new_Z = np.array(x_new_Z_)
            x_new_Z_ = x_new_Z.flatten()
            x_new_Z_mapped_scaled = np.array(x_new_Z_mapped_scaled_)
            y_new_Z = model_.predict(x_new_Z_mapped_scaled)  # Predict new y values based on updated model
            y_new_Z = y_new_Z.flatten()
            Eg_y = (35.37*x_new_Z_)/(y_new_Z*Temperature_R)
            Bg_y = 1/Eg_y
            Z_Eg = go.Figure(Z_Eg)
            Z_Eg.data[0].x = x_new_Z_
            Z_Eg.data[0].y =y_new_Z
            
            Z_Eg.data[2].x = x_new_Z_  
            Z_Eg.data[2].y = Bg_y
            
            Z_Eg.data[3].x = x_new_Z_  
            Z_Eg.data[3].y = Eg_y
            
            
            
            return Z_data, disable_indices_Z, fig_Z, popt,z_smooth_pressure_mapped_scaled,X_poly_mapped.tolist(),x_new_mapped_scaled.tolist(),df_r.to_dict('records'),Z_Eg,x_new_Z,x_new_Z_mapped_scaled
    

    return [{col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Z-Factor']}], [], go.Figure(), [],[],[],[],[{col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Z-Factor','Eg_Lab','Z-Factor_Smoothed','Eg','Gas Formation Volume Factor']}],go.Figure(),[],[]


# Solution_GOR

@callback(Output('output-table_Rsd', 'data'),
     Output('disable_indices_Rsd', 'data'),
     Output('Rsd_30_90', 'figure'),
     Output('popt_Rsd', 'data'),
     Input('submit-button_Rsd', 'n_clicks'),
     Input('clear-button_Rsd', 'n_clicks'),
     Input('Rsd_30_90', 'clickData'),
     State('output-table_d', 'data'),
     State('old-pb-input_d', 'value'),
     State('new-pb-input_d', 'value'),
     State('disable_indices_Rsd', 'data'),
     State('Rsd_30_90', 'figure'),
     State('popt_Rsd', 'data'),
     State('output-table_Rsd', 'data'),
     

    prevent_initial_call=True  # Ensure initial callback does not trigger
)
def update_store(submit_clicks, clear_clicks,click_rsd, current_data, old_Pb, New_Pb,disable_indices_rsd,fig_Rsd_30,popt,Rsd_old_data):
    def model_func_poly(x, a, b):
        return a * x + b
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'submit-button_Rsd':
        slope_Rsd, intercept_Rsd = None,None
        # Convert the data from the DataTable to a DataFrame
        df = pd.DataFrame(current_data)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
        df = df.applymap(
            lambda x: ''.join(filter(lambda ch: ch.isdigit() or ch in ['.', '-'], x)) if isinstance(x, str) else x)
        df = df.apply(pd.to_numeric, errors='coerce')
        df['Pressure(Psia)'] = df['Pressure(Psig)'] + 14.7
        Rsd_old= df[['Pressure(Psig)','Pressure(Psia)','Rsd']]
        Rsd_old_b = Rsd_old[Rsd_old['Pressure(Psig)']<=old_Pb]
        Rsd_old_30,Rsd_old_90 = Rsd_old_b['Pressure(Psig)'].iloc[(Rsd_old_b['Pressure(Psig)'] - old_Pb*0.3).abs().argmin()],Rsd_old_b['Pressure(Psig)'].iloc[(Rsd_old_b['Pressure(Psig)'] - old_Pb*0.9).abs().argmin()]
        Rsd_old_90_p = Rsd_old[(Rsd_old['Pressure(Psig)'] <=Rsd_old_90)] 
        Rsd_old_b_90 = Rsd_old[(Rsd_old['Pressure(Psig)'] <= old_Pb) & (Rsd_old['Pressure(Psig)'] >= Rsd_old_90)]
        Rsd_old_b_bet=Rsd_old_b_90.copy()
        Rsd_30_90= Rsd_old_b[(Rsd_old_b['Pressure(Psig)'] <= Rsd_old_90) & (Rsd_old_b['Pressure(Psig)'] >= Rsd_old_30)]
        x_30 = Rsd_30_90['Pressure(Psia)']
        y_30 = Rsd_30_90['Rsd']
        if old_Pb == New_Pb: 
           fig_Rsd_30,popt = go.Figure(),[] 
        else:
            fig_Rsd_30,popt=initial_plot('Pressure', 'Solution_GOR', x_30, y_30,'Solution_GOR vs. Pressure for Middle Part',model_func_poly)


        return Rsd_old.to_dict('records'),[],fig_Rsd_30,popt

    elif trigger_id == 'clear-button_Rsd':
        # Return an empty list to clear the stored data and DataTable
        return [{col: '' for col in ['Pressure(Psig)','Pressure(Psia)','Rsd']}],[],go.Figure(),[]

    elif trigger_id == 'Rsd_30_90':
        Rsd_old = pd.DataFrame(Rsd_old_data)
        Rsd_old_b = Rsd_old[Rsd_old['Pressure(Psig)'] <= old_Pb]
        fig_Rsd_30,disable_indices_rsd, popt=fitting_plot_1(click_rsd, fig_Rsd_30, disable_indices_rsd, model_func_poly)
        return Rsd_old.to_dict('records'), disable_indices_rsd, fig_Rsd_30,popt


    return [{col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Rsd']}],[], go.Figure(),[]


@callback(

    Output('disable_indices_Rsd_1', 'data'),
    Output('fig_Rsd', 'figure'),
    Output('popt_Rsd_1', 'data'),
    Output('scaler_poly_Rsd', 'data'),
    Output('x_mapped_Rsd', 'data'),
    Output('x_new_mapped_Rsd', 'data'),
    Output('result-table_Rsd', 'data'),
    Output('Rsd_Final', 'figure'),
    Output('x_new_Rsd', 'data'),
    Output('x_new_Rsd_mapped_scaled', 'data'),
    Output('Rsdb_new_S', 'data'),
    Input('submit-button_Rsd_1', 'n_clicks'),
    Input('clear-button_Rsd_1', 'n_clicks'),
    Input('fig_Rsd', 'clickData'),
    Input('Rsd-checklist', 'value'),  # Add checklist as input
    Input('Rsd-subchecklist', 'value'),  # Add sub-checklist as input
    State('output-table_Rsd', 'data'),
    State('old-pb-input_d', 'value'),
    State('new-pb-input_d', 'value'),
    State('disable_indices_Rsd_1', 'data'),
    State('fig_Rsd', 'figure'),
    State('popt_Rsd_1', 'data'),
    State('scaler_poly_Rsd', 'data'),
    State('x_mapped_Rsd', 'data'),
    State('x_new_mapped_Rsd', 'data'),
    State('result-table_Rsd', 'data'),
    State('Rsd_Final', 'figure'),
    State('x_new_Rsd', 'data'),
    State('x_new_Rsd_mapped_scaled', 'data'),
    State('popt_Rsd', 'data'),
    State('Rsdb_new_S', 'data'),
    prevent_initial_call=True
)
def update_store(submit_clicks, clear_clicks, click_Rsd, selected_option, selected_suboption, current_data, old_Pb,
                 New_Pb, disable_indices_Rsd_1, fig_Rsd, popt, Rsd_smooth_pressure_mapped_scaled,
                 x_mapped, x_new_mapped,result_data, Rsd_Final, x_new_Rsd_, x_new_Rsd_mapped_scaled_,popt_Rsd,Rsdb_new_S):

    

    def model_func_hyperbolic(x, a, b, c):
        return (a / (x + b)) + c

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id in ['submit-button_Rsd_1', 'Rsd-checklist', 'Rsd-subchecklist']:
        df = pd.DataFrame(current_data)
        Rsd_old = df[['Pressure(Psig)', 'Pressure(Psia)', 'Rsd']]
        Rsd_old_b = Rsd_old[Rsd_old['Pressure(Psig)'] <= old_Pb]
        
        if old_Pb == New_Pb:
            x_Rsd = Rsd_old_b['Pressure(Psia)']
            y_Rsd = Rsd_old_b['Rsd']
        else:
            slope = popt_Rsd[0]
            intercept = popt_Rsd[1]
            Rsd_old_30, Rsd_old_90 = Rsd_old_b['Pressure(Psig)'].iloc[
            (Rsd_old_b['Pressure(Psig)'] - old_Pb * 0.3).abs().argmin()], Rsd_old_b['Pressure(Psig)'].iloc[
            (Rsd_old_b['Pressure(Psig)'] - old_Pb * 0.9).abs().argmin()]
            Rsd_old_90_p = Rsd_old[(Rsd_old['Pressure(Psig)'] <= Rsd_old_90)]
            Rsd_old_b_90 = Rsd_old[(Rsd_old['Pressure(Psig)'] <= old_Pb) & (Rsd_old['Pressure(Psig)'] >= Rsd_old_90)]
            Rsd_old_b_bet = Rsd_old_b_90.copy()
            Rsd_30_90 = Rsd_old_b[
                (Rsd_old_b['Pressure(Psig)'] <= Rsd_old_90) & (Rsd_old_b['Pressure(Psig)'] >= Rsd_old_30)]
            Rsd_old_b_bet['Difference'] = Rsd_old_b_bet['Pressure(Psig)'].diff().abs().fillna(0)
            Rsd_old_b_bet['Cumulative_Sum'] = Rsd_old_b_bet['Difference'].cumsum()
            Rsd_old_b_bet['newpbsum'] = New_Pb - Rsd_old_b_bet['Cumulative_Sum']
            Rsd_old_b_bet['old_line'] = slope * Rsd_old_b_bet['Pressure(Psia)'] + intercept
            Rsd_old_b_bet['new_line'] = slope * (Rsd_old_b_bet['newpbsum'] + 14.7) + intercept
            Rsd_old_b_bet['New_Rsd'] = (Rsd_old_b_bet['Rsd'] - Rsd_old_b_bet['old_line']) + Rsd_old_b_bet['new_line']
            Rsd_old_b_I = Rsd_old_b_bet[['newpbsum', 'New_Rsd']].reset_index(drop=True)
            Rsd_old_b_I = Rsd_old_b_I.rename(columns={'newpbsum': 'Pressure(Psig)', 'New_Rsd': 'Rsd'})
            Rsd_old_b_I['Pressure(Psia)'] = Rsd_old_b_I['Pressure(Psig)'] + 14.7
            Rsd_old_b_I = Rsd_old_b_I[['Pressure(Psig)', 'Pressure(Psia)', 'Rsd']]
            Rsd_I = pd.concat([Rsd_old_b_I, Rsd_old_90_p], axis=0, ignore_index=True)
            x_Rsd = Rsd_I['Pressure(Psia)']
            y_Rsd = Rsd_I['Rsd']      
        # Model fitting functions based on checklist selection
        if 'Hyperbolic' in selected_option:
            model_func = model_func_hyperbolic      
            fig_Rsd, popt = initial_plot('Pressure', 'Solution GOR', x_Rsd, y_Rsd, 'Solution GOR vs. Pressure', model_func)
            result_df = Rsd_old.copy()
            new_pb_row = pd.DataFrame(
                {'Pressure(Psig)': [New_Pb], 'Pressure(Psia)': [New_Pb + 14.7], 'Rsd': [np.nan]})
            df_result = pd.concat([result_df, new_pb_row], ignore_index=True)
            df_result = df_result.drop_duplicates(subset='Pressure(Psig)', keep='first')
            df_result.sort_values(by='Pressure(Psig)', ascending=False, inplace=True)
            df_result_b=df_result[df_result['Pressure(Psig)'] <= New_Pb]
            Rsd_smooth_pressure = df_result_b['Pressure(Psia)'].values
            df_result_b['Rsd_Smoothed'] = model_func_hyperbolic(Rsd_smooth_pressure, *popt)
            matching_row = df_result_b.loc[df_result_b['Pressure(Psig)'] == New_Pb, 'Rsd_Smoothed']
            Rsdb_new_S = matching_row.iloc[0]
            x_new_Rsd = np.random.randint(np.min(Rsd_smooth_pressure), np.max(Rsd_smooth_pressure) + 1, 100)
            x_new_Rsd = np.sort(np.concatenate(([np.min(Rsd_smooth_pressure), np.max(Rsd_smooth_pressure)], x_new_Rsd)))
            y_new_Rsd = model_func_hyperbolic(x_new_Rsd, *popt)

            # Create the figure with two y-axes
            Rsd_Final = {
                'data': [
                    go.Scatter(x=x_new_Rsd, y=y_new_Rsd, mode='lines', name='Z-Factor_Smoothed',
                               customdata=np.arange(len(x_new_Rsd)), ),
                    go.Scatter(x=Rsd_old['Pressure(Psia)'].values, y=Rsd_old['Rsd'].values, mode='markers',marker=dict(color='red', size=10), name='Z-Factor_Smoothed',
                               customdata=np.arange(len(Rsd_old['Pressure(Psia)'].values)) )
                ],
                'layout': go.Layout(
                    title='Smoothed Solution GOR Vs. Pressure', xaxis={'title': 'Pressure (Psia)'},
                    yaxis={'title': 'Solution GOR'})}

            return [], fig_Rsd, popt, [], [], [], df_result_b.to_dict('records'), Rsd_Final, [], [],Rsdb_new_S

        elif 'Polynomial' in selected_option:
            degree = int(selected_suboption)
            # Update the plot with the selected model function
            fig_Rsd, popt, x_mapped, x_new_mapped, model, poly, scaler_poly = initial_plot_1('Pressure', 'Solution GOR', x_Rsd,y_Rsd, 'Z-Factor vs. Pressure',degree)
            result_df = Rsd_old.copy()
            new_pb_row = pd.DataFrame(
                {'Pressure(Psig)': [New_Pb], 'Pressure(Psia)': [New_Pb + 14.7], 'Rsd': [np.nan]})
            df_result = pd.concat([result_df, new_pb_row], ignore_index=True)
            df_result = df_result.drop_duplicates(subset='Pressure(Psig)', keep='first')
            df_result.sort_values(by='Pressure(Psig)', ascending=False, inplace=True)
            df_result_b=df_result[df_result['Pressure(Psig)'] <= New_Pb]
            Rsd_smooth_pressure = df_result_b['Pressure(Psia)'].values.reshape(-1, 1)
            Rsd_smooth_pressure_mapped = poly.transform(Rsd_smooth_pressure)
            Rsd_smooth_pressure_mapped_scaled = scaler_poly.transform(Rsd_smooth_pressure_mapped)
            Rsd_smooth_p = model.predict(Rsd_smooth_pressure_mapped_scaled)  # Predict new y values based on updated model
            Rsd_smooth_p = Rsd_smooth_p.flatten()
            df_result_b['Rsd_Smoothed']= Rsd_smooth_p


            x_new_Rsd = np.random.randint(np.min(df_result_b['Pressure(Psia)']), np.max(df_result_b['Pressure(Psia)']) + 1,
                                        100)
            x_new_Rsd = np.sort(
                np.concatenate(([np.min(df_result_b['Pressure(Psia)']), np.max(df_result_b['Pressure(Psia)'])], x_new_Rsd)))
            x_new_Rsd_ = x_new_Rsd.copy().flatten()
            x_new_Rsd = x_new_Rsd.reshape(-1, 1)
            x_new_Rsd_mapped = poly.transform(x_new_Rsd)
            x_new_Rsd_mapped_scaled = scaler_poly.transform(x_new_Rsd_mapped)
            y_new_Rsd = model.predict(x_new_Rsd_mapped_scaled)  # Predict new y values based on updated model
            y_new_Rsd = y_new_Rsd.flatten()
            matching_row = df_result_b.loc[df_result_b['Pressure(Psig)'] == New_Pb, 'Rsd_Smoothed']
            Rsdb_new_S = matching_row.iloc[0]
            Rsd_Final = {
                'data': [
                    go.Scatter(x=x_new_Rsd_, y=y_new_Rsd, mode='lines', name='Z-Factor_Smoothed',
                               customdata=np.arange(len(x_new_Rsd)), ),
                    go.Scatter(x=Rsd_old['Pressure(Psia)'].values, y=Rsd_old['Rsd'].values, mode='markers',marker=dict(color='red', size=10), name='Z-Factor_Smoothed',
                               customdata=np.arange(len(Rsd_old['Pressure(Psia)'].values)) )
                ],
                'layout': go.Layout(
                    title='Smoothed Solution GOR Vs. Pressure', xaxis={'title': 'Pressure (Psia)'},
                    yaxis={'title': 'Solution GOR'})}

            return [], fig_Rsd, popt, Rsd_smooth_pressure_mapped_scaled, x_mapped, x_new_mapped, df_result_b.to_dict(
                'records'), Rsd_Final, x_new_Rsd, x_new_Rsd_mapped_scaled,Rsdb_new_S

    elif trigger_id == 'clear-button_Z':
        return [], go.Figure(), [], [], [], [], [
            {col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Rsd', 'Rsd_Smoothed']}], go.Figure(), [], [],[]

    elif trigger_id == 'fig_Rsd':
        if 'Hyperbolic' in selected_option:
            fig_Rsd, disable_indices_Rsd_1, popt = fitting_plot_1(click_Rsd, fig_Rsd, disable_indices_Rsd_1, model_func_hyperbolic)
            df_r = pd.DataFrame(result_data)
            Rsd_smooth_pressure = df_r['Pressure(Psia)'].values
            df_r['Rsd_Smoothed'] = model_func_hyperbolic(Rsd_smooth_pressure, *popt)

            x_new_Rsd = np.random.randint(np.min(Rsd_smooth_pressure), np.max(Rsd_smooth_pressure) + 1, 100)
            x_new_Rsd = np.sort(np.concatenate(([np.min(Rsd_smooth_pressure), np.max(Rsd_smooth_pressure)], x_new_Rsd)))
            y_new_Rsd = model_func_hyperbolic(x_new_Rsd, *popt)
            matching_row = df_r.loc[df_r['Pressure(Psig)'] == New_Pb, 'Rsd_Smoothed']
            Rsdb_new_S = matching_row.iloc[0]

            Rsd_Final = go.Figure(Rsd_Final)
            Rsd_Final.data[0].x = x_new_Rsd
            Rsd_Final.data[0].y = y_new_Rsd


            return disable_indices_Rsd_1, fig_Rsd, popt, [], [], [], df_r.to_dict('records'), Rsd_Final, [], [],Rsdb_new_S

        elif 'Polynomial' in selected_option:
            X_poly_mapped = np.array(x_mapped)
            x_new_mapped_scaled = np.array(x_new_mapped)
            fig_Rsd, disable_indices_Rsd_1, popt, model_ = fitting_plot(click_Rsd, fig_Rsd, disable_indices_Rsd_1, X_poly_mapped,
                                                                  x_new_mapped_scaled)
            df_r = pd.DataFrame(result_data)
            Rsd_smooth_pressure_mapped_scaled_ = np.array(Rsd_smooth_pressure_mapped_scaled)
            Rsd_smooth_p = model_.predict(Rsd_smooth_pressure_mapped_scaled_)  # Predict new y values based on updated model
            Rsd_smooth_p = Rsd_smooth_p.flatten()
            df_r['Rsd_Smoothed'] = Rsd_smooth_p

            x_new_Rsd = np.array(x_new_Rsd_)
            x_new_Rsd_ = x_new_Rsd.flatten()
            x_new_Rsd_mapped_scaled = np.array(x_new_Rsd_mapped_scaled_)
            y_new_Rsd = model_.predict(x_new_Rsd_mapped_scaled)  # Predict new y values based on updated model
            y_new_Rsd = y_new_Rsd.flatten()

            Rsd_Final = go.Figure(Rsd_Final)
            Rsd_Final.data[0].x = x_new_Rsd_
            Rsd_Final.data[0].y = y_new_Rsd
            matching_row = df_r.loc[df_r['Pressure(Psig)'] == New_Pb, 'Rsd_Smoothed']
            Rsdb_new_S = matching_row.iloc[0]


            return disable_indices_Rsd_1, fig_Rsd, popt, Rsd_smooth_pressure_mapped_scaled, X_poly_mapped.tolist(), x_new_mapped_scaled.tolist(), df_r.to_dict(
                'records'), Rsd_Final, x_new_Rsd, x_new_Rsd_mapped_scaled,Rsdb_new_S

    return [], go.Figure(), [], [], [], [], [
        {col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Rsd', 'Rsd_Smoothed']}], go.Figure(), [], [],Rsdb_new_S


# Specific_Gravity

@callback(Output('output-table_SG', 'data'),
          Output('disable_indices_Rsd_SG', 'data'),
          Output('Rsd_SG', 'figure'),
          Output('popt_Rsd_SG', 'data'),
          Output('SG_df_Rsd_SG', 'data'),
          Output('x_mapped_SG', 'data'),
          Output('x_new_mapped_SG', 'data'),
          Output('Rsd_SG_smooth_pressure_mapped_scaled', 'data'),
          Output('Rsdb_old_S', 'data'),
          Input('submit-button_SG', 'n_clicks'),
          Input('clear-button_SG', 'n_clicks'),
          Input('Rsd_SG', 'clickData'),
          Input('SG-checklist_1', 'value'),  # Add checklist as input
          Input('SG-subchecklist_1', 'value'),
          State('output-table_d', 'data'),
          State('old-pb-input_d', 'value'),
          State('new-pb-input_d', 'value'),
          State('disable_indices_Rsd_SG', 'data'),
          State('Rsd_SG', 'figure'),
          State('popt_Rsd_SG', 'data'),
          State('output-table_SG', 'data'),
          State('result-table_Rsd', 'data'),
          State('SG_df_Rsd_SG', 'data'),
          State('x_mapped_SG', 'data'),
          State('x_new_mapped_SG', 'data'),
          State('Rsd_SG_smooth_pressure_mapped_scaled', 'data'),
          State('Rsdb_old_S', 'data'),

          prevent_initial_call=True  # Ensure initial callback does not trigger
          )
def update_store(submit_clicks, clear_clicks, click_SG,selected_option, selected_suboption, current_data, old_Pb, New_Pb, disable_indices_Rsd_SG, Rsd_SG,
                 popt, SG_data,smoothed_rsd_data,sgdf,x_mapped, x_new_mapped,Rsd_SG_smooth_pressure_mapped_scaled,Rsdb_old_S):
    def model_func_hyperbolic(x, a, b, c):
        return (a / (x + b)) + c

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id in ['submit-button_SG', 'SG-checklist_1', 'SG-subchecklist_1']:
        df_1 = pd.DataFrame(current_data)
        df_1 = df_1.replace(r'^\s*$', np.nan, regex=True)
        df_1 = df_1.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
        df_1 = df_1.applymap(
            lambda x: ''.join(filter(lambda ch: ch.isdigit() or ch in ['.', '-'], x)) if isinstance(x, str) else x)
        df_1 = df_1.apply(pd.to_numeric, errors='coerce')
        df_1['Pressure(Psia)'] = df_1['Pressure(Psig)'] + 14.7
        SG_old = df_1[['Pressure(Psig)', 'Pressure(Psia)','Rsd', 'SG']]
        SG_old_1 = SG_old[SG_old['Pressure(Psig)']<=old_Pb]
        SG_old_2 = SG_old[SG_old['Pressure(Psig)']<=old_Pb]

        if New_Pb==old_Pb:
            df = pd.DataFrame(smoothed_rsd_data)
            df_3 = SG_old_1[['SG']]
            df.reset_index(drop=True, inplace=True)
            df_3.reset_index(drop=True, inplace=True)
            df_4 = pd.concat([df, df_3], axis=1)
            df_4 = df_4[['Pressure(Psig)', 'Pressure(Psia)', 'Rsd', 'Rsd_Smoothed','SG']]
            matching_row = df_4.loc[df_4['Pressure(Psig)'] == old_Pb, 'Rsd_Smoothed']
            Rsdb_old_S = matching_row.iloc[0]
            Rsd_SG,popt = go.Figure(),[]
            return SG_old_1.to_dict('records'),[],Rsd_SG,popt,df_4.to_dict('records'),[],[],[],Rsdb_old_S

        else:
            x_Rsd_SG = SG_old_1['Pressure(Psia)']
            y_Rsd_SG = SG_old_1['Rsd']

            if 'Hyperbolic' in selected_option:
                model_func = model_func_hyperbolic
                Rsd_SG, popt = initial_plot('Pressure', 'Specific Gravity', x_Rsd_SG, y_Rsd_SG, 'Specific Gravity vs. Pressure', model_func)
                Rsd_SG_smooth_pressure = SG_old_2['Pressure(Psia)'].values
                SG_old_2['Rsd_Smoothed'] = model_func_hyperbolic(Rsd_SG_smooth_pressure, *popt)
                df_4 = SG_old_2[['Pressure(Psig)', 'Pressure(Psia)', 'Rsd', 'Rsd_Smoothed','SG']]
                matching_row = df_4.loc[df_4['Pressure(Psig)'] == old_Pb, 'Rsd_Smoothed']
                Rsdb_old_S = matching_row.iloc[0]

                return SG_old_1.to_dict('records'),[],Rsd_SG,popt,df_4.to_dict('records'),[],[],[],Rsdb_old_S

            elif 'Polynomial' in selected_option:
                degree = int(selected_suboption)
                # Update the plot with the selected model function
                Rsd_SG, popt, x_mapped, x_new_mapped, model, poly, scaler_poly = initial_plot_1('Pressure', 'Solution GOR', x_Rsd_SG,y_Rsd_SG, 'Solution GOR vs. Pressure',degree)
                
                Rsd_SG_smooth_pressure = SG_old_1['Pressure(Psia)'].values.reshape(-1, 1)
                Rsd_SG_smooth_pressure_mapped = poly.transform(Rsd_SG_smooth_pressure)
                Rsd_SG_smooth_pressure_mapped_scaled = scaler_poly.transform(Rsd_SG_smooth_pressure_mapped)
                Rsd_SG_smooth_p = model.predict(Rsd_SG_smooth_pressure_mapped_scaled)  # Predict new y values based on updated model
                Rsd_SG_smooth_p = Rsd_SG_smooth_p.flatten()
                SG_old_1['Rsd_Smoothed']= Rsd_SG_smooth_p
                df_4 = SG_old_1[['Pressure(Psig)', 'Pressure(Psia)', 'Rsd', 'Rsd_Smoothed','SG']]
                matching_row = df_4.loc[df_4['Pressure(Psig)'] == old_Pb, 'Rsd_Smoothed']
                Rsdb_old_S = matching_row.iloc[0]
                return SG_old_1.to_dict('records'),[],Rsd_SG,popt,df_4.to_dict('records'),x_mapped, x_new_mapped,Rsd_SG_smooth_pressure_mapped_scaled,Rsdb_old_S

    elif trigger_id == 'clear-button_Rsd':
        # Return an empty list to clear the stored data and DataTable
        return [{col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Rsd', 'SG']}], [], go.Figure(), [],[],[],[],[],[]

    elif trigger_id == 'Rsd_SG':
        if 'Hyperbolic' in selected_option:
            Rsd_SG, disable_indices_Rsd_SG, popt = fitting_plot_1(click_SG, Rsd_SG, disable_indices_Rsd_SG, model_func_hyperbolic)
            df_4 = pd.DataFrame(sgdf)
            SG_old_1 = pd.DataFrame(SG_data)
            Rsd_SG_smooth_pressure = df_4['Pressure(Psia)'].values
            df_4['Rsd_Smoothed'] = model_func_hyperbolic(Rsd_SG_smooth_pressure, *popt)
            matching_row = df_4.loc[df_4['Pressure(Psig)'] == old_Pb, 'Rsd_Smoothed']
            Rsdb_old_S = matching_row.iloc[0]
            return SG_old_1.to_dict('records'),disable_indices_Rsd_SG,Rsd_SG,popt,df_4.to_dict('records'),[],[],[],Rsdb_old_S

        elif 'Polynomial' in selected_option:
            X_poly_mapped = np.array(x_mapped)
            x_new_mapped_scaled = np.array(x_new_mapped)
            Rsd_SG, disable_indices_Rsd_SG, popt, model_ = fitting_plot(click_SG, Rsd_SG, disable_indices_Rsd_SG, X_poly_mapped,x_new_mapped_scaled)
            SG_old_1 = pd.DataFrame(SG_data)                                                     
            df_4 = pd.DataFrame(sgdf)
            Rsd_SG_smooth_pressure_mapped_scaled_ = np.array(Rsd_SG_smooth_pressure_mapped_scaled)
            Rsd_SG_smooth_p = model_.predict(Rsd_SG_smooth_pressure_mapped_scaled_)  # Predict new y values based on updated model
            Rsd_SG_smooth_p = Rsd_SG_smooth_p.flatten()
            df_4['Rsd_Smoothed'] = Rsd_SG_smooth_p
            matching_row = df_4.loc[df_4['Pressure(Psig)'] == old_Pb, 'Rsd_Smoothed']
            Rsdb_old_S = matching_row.iloc[0]
            return SG_old_1.to_dict('records'),disable_indices_Rsd_SG,Rsd_SG,popt,df_4.to_dict('records'),X_poly_mapped.tolist(), x_new_mapped_scaled.tolist(),Rsd_SG_smooth_pressure_mapped_scaled,Rsdb_old_S

    return [{col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'Rsd', 'SG']}], [], go.Figure(), [],[],[],[],[],Rsdb_old_S

@callback(

    Output('disable_indices_SG', 'data'),
    Output('fig_SG', 'figure'),
    Output('popt_SG', 'data'),
    Output('scaler_poly_SG', 'data'),
    Output('x_mapped_SG_1', 'data'),
    Output('x_new_mapped_SG_1', 'data'),
    Output('result-table_SG', 'data'),
    Output('SG_Final', 'figure'),
    Output('x_new_SG', 'data'),
    Output('x_new_SG_mapped_scaled', 'data'),
    Output('df_7', 'data'),
    Output('Rsd_SG_Density', 'data'),
    Input('submit-button_SG_1', 'n_clicks'),
    Input('clear-button_SG_1', 'n_clicks'),
    Input('fig_SG', 'clickData'),
    Input('SG-checklist', 'value'),  # Add checklist as input
    Input('SG-subchecklist', 'value'),  # Add sub-checklist as input
    State('SG_df_Rsd_SG', 'data'),
    State('old-pb-input_d', 'value'),
    State('new-pb-input_d', 'value'),
    State('disable_indices_SG', 'data'),
    State('fig_SG', 'figure'),
    State('popt_SG', 'data'),
    State('scaler_poly_SG', 'data'),
    State('x_mapped_SG_1', 'data'),
    State('x_new_mapped_SG_1', 'data'),
    State('result-table_SG', 'data'),
    State('SG_Final', 'figure'),
    State('x_new_SG', 'data'),
    State('x_new_SG_mapped_scaled', 'data'),
    State('output-table_d', 'data'),
    State('result-table_Rsd', 'data'),
    State('df_7', 'data'),
    State('Rsd_SG_Density', 'data'),
    prevent_initial_call=True
)
def update_store(submit_clicks, clear_clicks, click_SG, selected_option, selected_suboption, current_data, old_Pb,
                 New_Pb, disable_indices_SG, fig_SG, popt, SG_smooth_pressure_mapped_scaled,
                 x_mapped, x_new_mapped,result_data, SG_Final, x_new_SG_, x_new_SG_mapped_scaled_,differntial_data,gor_data,df_7,Rsd_SG_Density):
    def model_func_hyperbolic(x, a, b, c):
        return (a / (x + b)) + c
    def model_func_exp(x, a, b, c, d):
        return a * (b ** (c * x)) + d
    def model_func_log(x, a, b):
        return a * np.log(x) + b
    def model_func_power(x, a, b):
        return a * (x ** b)
    def modified_hyperbolic(x, a, b, c, d):
        return  (a / (x + b)) + c * (x ** d)
    def Decline_hyperbolic(x,a,b,c,d):
        return a / ((c*d*x + b)**(1/c))


    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id in ['submit-button_SG_1', 'SG-checklist', 'SG-subchecklist']:
        df = pd.DataFrame(current_data)
        df_1 =df.copy()
        Rsd_Density = df_1[['Pressure(Psig)', 'Pressure(Psia)','Rsd_Smoothed']]
        ind = df[df['Pressure(Psig)'] == old_Pb].index[0]
        Rsd_old_pb = df.loc[ind, 'Rsd']
        df['Rsd_d'] = Rsd_old_pb - df['Rsd']
        df['Rsd_d_d'] = df['Rsd_d'].diff()
        df['SG_Cum'] = 0
        df = df.fillna(0)
        for i in range(1, len(df)):
            df['SG_Cum'].iloc[i] = (df['Rsd_d_d'].iloc[i] * df['SG'].iloc[i] + df['SG_Cum'].iloc[i - 1] * df['Rsd_d'].iloc[
                i - 1]) / df['Rsd_d'].iloc[i]
        Rsd_old_pb_s = df.loc[ind, 'Rsd_Smoothed']
        df['Rsd_d_s'] = Rsd_old_pb_s - df['Rsd_Smoothed']
        df['Rsd_d_d_s'] = df['Rsd_d_s'].diff()
        df['SG_Corrected'] = (df['SG_Cum'] * df['Rsd_d_s'] - df['SG_Cum'].shift(1) * df['Rsd_d_s'].shift(1)) / (df['Rsd_d_d_s'])
        df_2 = df[df['Pressure(Psig)'] <old_Pb]
        df_3 = pd.DataFrame(differntial_data)
        df_6 = pd.DataFrame(gor_data)
        df_6 = df_6[['Rsd_Smoothed']]
        df_6.rename(columns={'Rsd_Smoothed': 'Rsd_Smoothed_1'}, inplace=True)
        df_3 = df_3[['Pressure(Psig)','Pressure(Psia)']]
        df_4 = df_3[df_3['Pressure(Psig)'] > old_Pb]
        result_df = df[['Pressure(Psig)', 'Pressure(Psia)', 'SG', 'SG_Corrected', 'SG_Cum']]

        new_pb_row = pd.DataFrame(
            {'Pressure(Psig)': [New_Pb], 'Pressure(Psia)': [New_Pb + 14.7], 'SG': [np.nan], 'SG_Corrected': [np.nan],
             'SG_Cum': [np.nan]})
        df_result = pd.concat([df_4, result_df, new_pb_row], ignore_index=True)
        df_result = df_result.drop_duplicates(subset='Pressure(Psig)', keep='first')
        df_result.sort_values(by='Pressure(Psig)', ascending=False, inplace=True)
        df_result_b = df_result[df_result['Pressure(Psig)'] <= New_Pb]
        df_7 = df_result_b.copy()
        df_7.reset_index(drop=True, inplace=True)
        df_6.reset_index(drop=True, inplace=True)
        df_7 = pd.concat([df_7, df_6], axis=1)
        ind_1 = df_7[df_7['Pressure(Psig)'] == New_Pb].index[0]
        Rsd_New_pb = df_7.loc[ind_1, 'Rsd_Smoothed_1']
        df_7['Rsd_d_1'] = Rsd_New_pb - df_7['Rsd_Smoothed_1']
        df_7['Rsd_d_d_1'] = df_7['Rsd_d_1'].diff()
        df_7['SG_Cum_Smoothed'] = 0
        df_7 = df_7.fillna(0)
        x_SG = df_2['Pressure(Psia)']
        y_SG = df_2['SG_Corrected']
        # Model fitting functions based on checklist selection
        if 'Poly' not in selected_option:
            model_func = model_func_hyperbolic
            if 'H(A)' in selected_option:
                model_func = model_func_hyperbolic
            elif 'Exp' in selected_option:
                model_func = model_func_exp
            elif 'Log' in selected_option:
                model_func = model_func_log
            elif 'Power' in selected_option:
                model_func = model_func_power
            elif 'H(B)' in selected_option:
                model_func = modified_hyperbolic
            elif 'H(C)' in selected_option:
                model_func = Decline_hyperbolic

            fig_SG, popt = initial_plot('Pressure', 'Specific Gravity', x_SG, y_SG, 'Specific Gravity vs. Pressure', model_func)
            SG_smooth_pressure = df_result_b['Pressure(Psia)'].values
            df_7['SG_Smoothed'] = model_func(SG_smooth_pressure, *popt)
            for i in range(1, len(df_7)):
                df_7['SG_Cum_Smoothed'].iloc[i] = (df_7['Rsd_d_d_1'].iloc[i] * df_7['SG_Smoothed'].iloc[i] +
                                                   df_7['SG_Cum_Smoothed'].iloc[i - 1] * df_7['Rsd_d_1'].iloc[
                                                       i - 1]) / df_7['Rsd_d_1'].iloc[i]
            df_result_b = pd.merge(df_result_b, df_7[['Pressure(Psig)', 'SG_Smoothed', 'SG_Cum_Smoothed']], on='Pressure(Psig)', how='left')
            SG_Density=df_result_b[['Pressure(Psig)','SG_Smoothed']]
            SG_Density = SG_Density[SG_Density['Pressure(Psig)']<=old_Pb]
            SG_Density = SG_Density[['SG_Smoothed']]
            Rsd_Density = Rsd_Density.reset_index(drop=True)
            SG_Density= SG_Density.reset_index(drop=True)
            Rsd_SG_Density= pd.concat([Rsd_Density, SG_Density], axis=1)
            x_new_SG = np.random.randint(np.min(SG_smooth_pressure), np.max(SG_smooth_pressure) + 1, 300)
            x_new_SG = np.sort(np.concatenate(([np.min(SG_smooth_pressure), np.max(SG_smooth_pressure)], x_new_SG)))
            y_new_SG = model_func(x_new_SG, *popt)
            df_result_b_cum = df_result_b[df_result_b['Pressure(Psig)'] <New_Pb]

            # Create the figure with two y-axes
            SG_Final = {
                'data': [
                    go.Scatter(x=x_new_SG, y=y_new_SG, mode='lines', name='SG_Smoothed',
                               customdata=np.arange(len(x_new_SG)), ),
                    go.Scatter(x=df_2['Pressure(Psia)'].values, y=df_2['SG'].values, mode='markers',marker=dict(color='blue', size=10), name='SG',
                               customdata=np.arange(len(df_2['Pressure(Psia)'].values)) ),
                    go.Scatter(x=df_result_b_cum['Pressure(Psia)'].values, y=df_result_b_cum['SG_Cum_Smoothed'].values, mode='lines', name='SG_Cum_Smoothed',
                               customdata=np.arange(len(df_result_b_cum['Pressure(Psia)'].values)),yaxis='y2' ),

                ],
                'layout': go.Layout(
                    title='Smoothed Specific Gravity Vs. Pressure', xaxis={'title': 'Pressure (Psia)'},
                    yaxis={'title': 'Specific Gravity'},yaxis2={'title': 'Cum_SG','overlaying': 'y', 'side': 'right'  })}

            return [], fig_SG, popt, [], [], [], df_result_b.to_dict('records'), SG_Final, [], [],df_7.to_dict('records'),Rsd_SG_Density.to_dict('records')

        elif 'Poly' in selected_option:
            degree = int(selected_suboption)
            # Update the plot with the selected model function
            fig_SG, popt, x_mapped, x_new_mapped, model, poly, scaler_poly = initial_plot_1('Pressure', 'Specific Gravity', x_SG,y_SG, 'Specific Gravity vs. Pressure',degree)


            SG_smooth_pressure = df_result_b['Pressure(Psia)'].values.reshape(-1, 1)
            SG_smooth_pressure_mapped = poly.transform(SG_smooth_pressure)
            SG_smooth_pressure_mapped_scaled = scaler_poly.transform(SG_smooth_pressure_mapped)
            SG_smooth_p = model.predict(SG_smooth_pressure_mapped_scaled)  # Predict new y values based on updated model
            SG_smooth_p = SG_smooth_p.flatten()
            df_7['SG_Smoothed'] = SG_smooth_p
            for i in range(1, len(df_7)):
                df_7['SG_Cum_Smoothed'].iloc[i] = (df_7['Rsd_d_d_1'].iloc[i] * df_7['SG_Smoothed'].iloc[i] +
                                                   df_7['SG_Cum_Smoothed'].iloc[i - 1] * df_7['Rsd_d_1'].iloc[
                                                       i - 1]) / df_7['Rsd_d_1'].iloc[i]
            df_result_b = pd.merge(df_result_b, df_7[['Pressure(Psig)', 'SG_Smoothed', 'SG_Cum_Smoothed']], on='Pressure(Psig)', how='left')
            SG_Density = df_result_b[['Pressure(Psig)', 'SG_Smoothed']]
            SG_Density = SG_Density[SG_Density['Pressure(Psig)'] <= old_Pb]
            SG_Density = SG_Density[['SG_Smoothed']]
            Rsd_Density = Rsd_Density.reset_index(drop=True)
            SG_Density = SG_Density.reset_index(drop=True)
            Rsd_SG_Density = pd.concat([Rsd_Density, SG_Density], axis=1)
            x_new_SG = np.random.randint(np.min(df_result_b['Pressure(Psia)']), np.max(df_result_b['Pressure(Psia)']) + 1,
                                        300)
            x_new_SG = np.sort(
                np.concatenate(([np.min(df_result_b['Pressure(Psia)']), np.max(df_result_b['Pressure(Psia)'])], x_new_SG)))
            x_new_SG_ = x_new_SG.copy().flatten()
            x_new_SG = x_new_SG.reshape(-1, 1)
            x_new_SG_mapped = poly.transform(x_new_SG)
            x_new_SG_mapped_scaled = scaler_poly.transform(x_new_SG_mapped)
            y_new_SG = model.predict(x_new_SG_mapped_scaled)  # Predict new y values based on updated model
            y_new_SG = y_new_SG.flatten()
            df_result_b_cum = df_result_b[df_result_b['Pressure(Psig)'] < New_Pb]
            SG_Final = {
                'data': [
                    go.Scatter(x=x_new_SG_, y=y_new_SG, mode='lines', name='SG_Smoothed',
                               customdata=np.arange(len(x_new_SG)), ),
                    go.Scatter(x=df_2['Pressure(Psia)'].values, y=df_2['SG'].values, mode='markers',
                               marker=dict(color='red', size=10), name='SG',
                               customdata=np.arange(len(df_2['Pressure(Psia)'].values))),
                    go.Scatter(x=df_result_b_cum['Pressure(Psia)'].values, y=df_result_b_cum['SG_Cum_Smoothed'].values, mode='lines',
                               name='SG_Cum_Smoothed',
                               customdata=np.arange(len(df_result_b_cum['Pressure(Psia)'].values)),yaxis='y2' )
                ],
                'layout': go.Layout(
                    title='Smoothed Specific Gravity Vs. Pressure', xaxis={'title': 'Pressure (Psia)'},
                    yaxis={'title': 'Specific Gravity'},yaxis2={'title': 'Cum_SG','overlaying': 'y', 'side': 'right'  })}

            return [], fig_SG, popt, SG_smooth_pressure_mapped_scaled, x_mapped, x_new_mapped, df_result_b.to_dict(
                'records'), SG_Final, x_new_SG, x_new_SG_mapped_scaled,df_7.to_dict('records'),Rsd_SG_Density.to_dict('records')

    elif trigger_id == 'clear-button_Z':
        return [], go.Figure(), [], [], [], [], [
            {col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'SG', 'SG_Corrected', 'SG_Cum','SG_Smoothed','SG_Cum_Smoothed']}], go.Figure(), [], [],[],[]

    elif trigger_id == 'fig_SG':
        if 'Poly' not in selected_option:
            if 'H(A)' in selected_option:
                model_func = model_func_hyperbolic
            elif 'Exp' in selected_option:
                model_func = model_func_exp
            elif 'Log' in selected_option:
                model_func = model_func_log
            elif 'Power' in selected_option:
                model_func = model_func_power
            elif 'H(B)' in selected_option:
                model_func = modified_hyperbolic
            elif 'H(C)' in selected_option:
                model_func = Decline_hyperbolic
            fig_SG, disable_indices_SG, popt = fitting_plot_1(click_SG, fig_SG, disable_indices_SG, model_func)
            df_r = pd.DataFrame(result_data)
            df_7 = pd.DataFrame(df_7)
            SG_smooth_pressure = df_r['Pressure(Psia)'].values
            df_7['SG_Smoothed'] = model_func(SG_smooth_pressure, *popt)
            for i in range(1, len(df_7)):
                df_7['SG_Cum_Smoothed'].iloc[i] = (df_7['Rsd_d_d_1'].iloc[i] * df_7['SG_Smoothed'].iloc[i] +
                                                   df_7['SG_Cum_Smoothed'].iloc[i - 1] * df_7['Rsd_d_1'].iloc[
                                                       i - 1]) / df_7['Rsd_d_1'].iloc[i]
            df_r.drop(columns=['SG_Smoothed', 'SG_Cum_Smoothed'], inplace=True)
            df_r = pd.merge(df_r, df_7[['Pressure(Psig)', 'SG_Smoothed', 'SG_Cum_Smoothed']], on='Pressure(Psig)', how='left')
            Rsd_Density = df_7[['Pressure(Psig','Pressure(Psia)','Rsd_Smoothed']]
            Rsd_Density = Rsd_Density[Rsd_Density['Pressure(Psig)'] <= old_Pb]
            SG_Density = df_r[['Pressure(Psig)', 'SG_Smoothed']]
            SG_Density = SG_Density[SG_Density['Pressure(Psig)'] <= old_Pb]
            SG_Density = SG_Density[['SG_Smoothed']]
            Rsd_Density = Rsd_Density.reset_index(drop=True)
            SG_Density = SG_Density.reset_index(drop=True)
            Rsd_SG_Density = pd.concat([Rsd_Density, SG_Density], axis=1)
            x_new_SG = np.random.randint(np.min(SG_smooth_pressure), np.max(SG_smooth_pressure) + 1, 300)
            x_new_SG = np.sort(np.concatenate(([np.min(SG_smooth_pressure), np.max(SG_smooth_pressure)], x_new_SG)))
            y_new_SG = model_func(x_new_SG, *popt)
            df_result_b_cum = df_r[df_r['Pressure(Psig)'] < New_Pb]
            print(df_result_b_cum)
            SG_Final = go.Figure(SG_Final)
            SG_Final.data[0].x = x_new_SG
            SG_Final.data[0].y = y_new_SG
            SG_Final.data[2].x = df_result_b_cum['Pressure(Psia)'].values
            SG_Final.data[2].y = df_result_b_cum['SG_Cum_Smoothed'].values


            return disable_indices_SG, fig_SG, popt, [], [], [], df_r.to_dict('records'), SG_Final, [], [],df_7.to_dict('records'),Rsd_SG_Density.to_dict('records')

        elif 'Poly' in selected_option:
            X_poly_mapped = np.array(x_mapped)
            x_new_mapped_scaled = np.array(x_new_mapped)
            fig_SG, disable_indices_SG, popt, model_ = fitting_plot(click_SG, fig_SG, disable_indices_SG, X_poly_mapped,x_new_mapped_scaled)
            df_r = pd.DataFrame(result_data)
            df_7 = pd.DataFrame(df_7)
            SG_smooth_pressure_mapped_scaled_ = np.array(SG_smooth_pressure_mapped_scaled)
            SG_smooth_p = model_.predict(SG_smooth_pressure_mapped_scaled_)  # Predict new y values based on updated model
            SG_smooth_p = SG_smooth_p.flatten()
            df_7['SG_Smoothed'] = SG_smooth_p
            for i in range(1, len(df_7)):
                df_7['SG_Cum_Smoothed'].iloc[i] = (df_7['Rsd_d_d_1'].iloc[i] * df_7['SG_Smoothed'].iloc[i] +
                                                   df_7['SG_Cum_Smoothed'].iloc[i - 1] * df_7['Rsd_d_1'].iloc[
                                                       i - 1]) / df_7['Rsd_d_1'].iloc[i]
            df_r.drop(columns=['SG_Smoothed', 'SG_Cum_Smoothed'], inplace=True)
            df_r = pd.merge(df_r, df_7[['Pressure(Psig)', 'SG_Smoothed', 'SG_Cum_Smoothed']], on='Pressure(Psig)', how='left')
            Rsd_Density = df_7[['Pressure(Psig', 'Pressure(Psia)', 'Rsd_Smoothed']]
            Rsd_Density = Rsd_Density[Rsd_Density['Pressure(Psig)'] <= old_Pb]
            SG_Density = df_r[['Pressure(Psig)', 'SG_Smoothed']]
            SG_Density = SG_Density[SG_Density['Pressure(Psig)'] <= old_Pb]
            SG_Density = SG_Density[['SG_Smoothed']]
            Rsd_Density = Rsd_Density.reset_index(drop=True)
            SG_Density = SG_Density.reset_index(drop=True)
            Rsd_SG_Density = pd.concat([Rsd_Density, SG_Density], axis=1)
            x_new_SG = np.array(x_new_SG_)
            x_new_SG_ = x_new_SG.flatten()
            x_new_SG_mapped_scaled = np.array(x_new_SG_mapped_scaled_)
            y_new_SG = model_.predict(x_new_SG_mapped_scaled)  # Predict new y values based on updated model
            y_new_SG = y_new_SG.flatten()
            df_result_b_cum = df_r[df_r['Pressure(Psig)'] < New_Pb]
            SG_Final = go.Figure(SG_Final)
            SG_Final.data[0].x = x_new_SG_
            SG_Final.data[0].y = y_new_SG
            SG_Final.data[2].x = df_result_b_cum['Pressure(Psia)'].values
            SG_Final.data[2].y = df_result_b_cum['SG_Cum_Smoothed'].values


            return disable_indices_SG, fig_SG, popt, SG_smooth_pressure_mapped_scaled, X_poly_mapped.tolist(), x_new_mapped_scaled.tolist(), df_r.to_dict(
                'records'), SG_Final, x_new_SG, x_new_SG_mapped_scaled,df_7.to_dict('records'),Rsd_SG_Density.to_dict('records')

    return [], go.Figure(), [], [], [], [], [
            {col: '' for col in ['Pressure(Psig)', 'Pressure(Psia)', 'SG', 'SG_Corrected', 'SG_Cum','SG_Smoothed','SG_Cum_Smoothed']}], go.Figure(), [], [],[],[]

# Density

@app.callback(
 Output('output-table_Density', 'data'),
        Input('Prev', 'n_clicks'),
        Input('Fresh', 'n_clicks'),
        Input('submit-button_Density', 'n_clicks'),
        Input('clear-button_Density', 'n_clicks'),
        State('Bod_Density', 'data'),
        State('Rsd_SG_Density', 'data'),
        State('output-table_Density', 'data'),
        State('SG', 'value'),
        prevent_initial_call=True
                )
def update_table(prev_clicks, fresh_clicks,submit_clicks,clear_clicks, Bod_data,Rsd_data, current_data,SG_differntial):
    ctx = dash.callback_context

    if not ctx.triggered:
        return current_data

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'Prev':
        df = pd.DataFrame(Bod_data)
        df1 =  pd.DataFrame(Rsd_data)
        df1.drop(columns = 'Pressure(Psia)',inplace = True)
        merged_df = pd.merge(df, df1, on='Pressure(Psig)', how='left')
        print(merged_df)
        merged_df['Rsd_Diff'] = np.abs(merged_df['Rsd_Smoothed'].diff())
        merged_df['Gas_Volume'] = merged_df['Rsd_Diff'].fillna(0)
        merged_df['Gas_Mass'] = merged_df['Gas_Volume'] * merged_df['SG_Smoothed'] * 1.225
        merged_df['Gas_Mass'] = merged_df['Gas_Mass'].fillna(0)
        merged_df['Oil+Gas_Mass'] = float('nan')
        merged_df.loc[df.index[-1], 'Oil+Gas_Mass'] = 1000 * SG_differntial
        for i in range(len(merged_df) - 2, -1, -1):
            merged_df.loc[i, 'Oil+Gas_Mass'] = merged_df.loc[i + 1, 'Oil+Gas_Mass'] + merged_df.loc[i + 1, 'Gas_Mass']

        merged_df['Oil_Density_Calculated'] = merged_df['Oil+Gas_Mass'] / merged_df['Bod_Old'] / 1000
        merged_df=merged_df[['Pressure(Psig)', 'Pressure(Psia)', 'Bod_Old', 'Rsd_Smoothed', 'SG_Smoothed', 'Oil_Density_Calculated']]
        return merged_df.to_dict('records')

    elif button_id == 'Fresh':
        # Reset table for fresh data input
        return [{'Pressure(Psig)': '', 'Pressure(Psia)': '','Bod_Old':'','Rsd_Smoothed':'','SG_Smoothed':'','Oil_Density_Calculated':''} for _ in range(12)]
    elif button_id == 'submit-button_Density':
        df = pd.DataFrame(current_data)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

        # Remove any non-numeric characters (except decimal point and negative sign)
        df = df.applymap(
            lambda x: ''.join(filter(lambda ch: ch.isdigit() or ch in ['.', '-'], x)) if isinstance(x, str) else x)

        # Convert columns to float, handling errors by coercing invalid values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        merged_df = df.copy()
        merged_df['Rsd_Diff'] = np.abs(merged_df['Rsd_Smoothed'].diff())
        merged_df['Gas_Volume'] = merged_df['Rsd_Diff'].fillna(0)
        merged_df['Gas_Mass'] = merged_df['Gas_Volume'] * merged_df['SG_Smoothed'] * 1.225
        merged_df['Gas_Mass'] = merged_df['Gas_Mass'].fillna(0)
        merged_df['Oil+Gas_Mass'] = float('nan')
        merged_df.loc[merged_df.index[-1], 'Oil+Gas_Mass'] = 1000 * SG_differntial
        for i in range(len(merged_df) - 2, -1, -1):
            merged_df.loc[i, 'Oil+Gas_Mass'] = merged_df.loc[i + 1, 'Oil+Gas_Mass'] + merged_df.loc[i + 1, 'Gas_Mass']

        merged_df['Oil_Density_Calculated'] = merged_df['Oil+Gas_Mass'] / merged_df['Bod_Old'] / 1000
        merged_df = merged_df[
            ['Pressure(Psig)', 'Pressure(Psia)', 'Bod_Old', 'Rsd_Smoothed', 'SG_Smoothed', 'Oil_Density_Calculated']]
        return merged_df.to_dict('records')


    elif button_id == 'clear-button_Density':
        return [{'Pressure(Psig)': '', 'Pressure(Psia)': '', 'Bod_Old': '', 'Rsd_Smoothed': '', 'SG_Smoothed': '',
                 'Oil_Density_Calculated': ''} for _ in range(12)]


    return current_data  # Default: return the current table data

@app.callback(
        Output('factor-slider', 'value'),
        Output('factor-input', 'value'),
        Output('Density_corrected', 'figure'),
        Output('Density_Corrected_Bod', 'figure'),
        Output('output-corrected_Density', 'data'),
        Output('output-measured_Density', 'data'),
        Output('Bodb_old_S', 'data'),
        Output('Density_S', 'data'),
        Input('factor-slider', 'value'),
        Input('factor-input', 'value'),
        Input('submit-button_Density_measured', 'n_clicks'),
        Input('clear-button_Density_measured', 'n_clicks'),
        State('factor-slider', 'value'), 
        State('factor-input', 'value'),
        State('output-measured_Density', 'data'),
        State('output-table_Density', 'data'),
        State('old-pb-input_d', 'value'),
        State('new-pb-input_d', 'value'),
        State('Bodb_old_S', 'data'),
        State('Density_S', 'data'), 
        prevent_initial_call=True
                )
def update_table(slider_value, input_value, submit_clicks, clear_clicks, current_slider, current_input,measured_df,table_df,old_pb,new_pb,Bodb_old_S,Density_S):
    ctx = dash.callback_context

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'clear-button_Density_measured':
        # Clear the plot and table
        return 1,1,go.Figure(),go.Figure(),[{'name': col, 'id': col} for col in ['Pressure(Psig)','Corrected_Density','Corrected_Bod']],[{'Pressure': '', 'Measured_Density':''} for _ in range(12)],[],[] 

    elif trigger_id == 'submit-button':
        # Initial plot generation when submit button is clicked
        factor = slider_value
    else:
        # Update the plot after submit button is clicked
        factor = slider_value if trigger_id == 'factor-slider' else input_value

    measured_density = pd.DataFrame(measured_df)
    calculated_density = pd.DataFrame(table_df)
    
    
    calculated_density['calculated_factor'] = calculated_density['Bod_Old']*calculated_density['Oil_Density_Calculated']
    corrected_density_df = pd.DataFrame()
    corrected_density_df['Pressure(Psig)'] = calculated_density['Pressure(Psig)']
    corrected_density_df['Corrected_Density']=np.nan
    corrected_density_df['Corrected_Bod']=np.nan
    # Apply factor to calculated density
    corrected_density_df['Corrected_Density'] = calculated_density['Oil_Density_Calculated'] * factor
    corrected_density_df['Corrected_Bod']= calculated_density['calculated_factor']/corrected_density_df['Corrected_Density']
    x = corrected_density_df['Pressure(Psig)'].values
    y = corrected_density_df['Corrected_Density'].values
    y1 = measured_density['Measured_Density'][~measured_density['Measured_Density'].isna()].values
    x1 = measured_density['Pressure(Psig)'][~measured_density['Pressure(Psig)'].isna()].values

    x2 = corrected_density_df['Pressure(Psig)'].values
    y2 = corrected_density_df['Corrected_Bod'].values
    x3 = calculated_density['Pressure(Psig)'].values
    y3 = calculated_density['Bod_Old'].values

    matching_row = corrected_density_df.loc[corrected_density_df['Pressure(Psig)'] == old_pb, 'Corrected_Bod']
    Bodb_old_S = matching_row.iloc[0]
    matching_row = corrected_density_df.loc[corrected_density_df['Pressure(Psig)'] == old_pb, 'Corrected_Density']
    Density_S = matching_row.iloc[0]


    fig_density = {
                'data': [go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(color='blue', size=8), name='Corrected_Density',
                                    customdata=np.arange(len(x)), ),
                         go.Scatter(x=x1, y=y1, mode='lines+markers', marker=dict(color='red', size=8),
                                    name='Measured_Density')],
                'layout': go.Layout(title='Corrected_Density vs. Measured_Density', xaxis={'title': 'Pressure'},
                                    yaxis={'title': 'Density'},
                                    )}
    
    fig_Bod = {
                'data': [go.Scatter(x=x2, y=y2, mode='lines+markers', marker=dict(color='blue', size=8), name='Corrected_Bod',
                                    customdata=np.arange(len(x)), ),
                         go.Scatter(x=x3, y=y3, mode='lines+markers', marker=dict(color='red', size=8),
                                    name='Old_Bod')],
                'layout': go.Layout(title='Density Corrected Bod vs. Old_Bod', xaxis={'title': 'Pressure'},
                                    yaxis={'title': 'Oil Formation Volume Factor'},
                                    )}
   

    return factor, factor, fig_density, fig_Bod ,corrected_density_df.to_dict('records'),measured_density.to_dict('records'),Bodb_old_S,Density_S
@app.callback(
        Output('output-table_Extended_Density', 'data'),
        Output('Extended_Density', 'figure'),
        Output('Bodb_new_S', 'data'),
        Input('submit-button_Extended', 'n_clicks'),
        Input('clear-button_Extended', 'n_clicks'),
        Input('store-data_Vrel', 'data'),
        State('SG', 'value'),
        State('result-table_Bod', 'data'),
        State('result-table_Rsd', 'data'),
        State('result-table_SG', 'data'),
        State('output-corrected_Density', 'data'),
        State('old-pb-input_d', 'value'),
        State('new-pb-input_d', 'value'),
        State('output-table_Extended_Density', 'data'),
        State('Bodb_new_S', 'data'),
        prevent_initial_call=True
                )
def update_table(submit_clicks, clear_clicks,CME_Vrel,SG,Bod_result,Rsd_result,SG_result,Density_result, old_pb,new_pb,current_data,Bodb_new_S):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_data

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'submit-button_Extended':
        slope_CME = CME_Vrel.get('data1', 'No data for data1')
        df_Bod = pd.DataFrame(Bod_result)
        df_Rsd = pd.DataFrame(Rsd_result)
        df_SG = pd.DataFrame(SG_result)
        df_Density = pd.DataFrame(Density_result)
        df = df_Bod[['Pressure(Psig)','Pressure(Psia)','Vrel_Smoothed']]
        df_1 = pd.merge(df, df_Rsd[['Pressure(Psig)', 'Rsd_Smoothed']], on='Pressure(Psig)', how='left')
        df_2= pd.merge(df_1, df_SG[['Pressure(Psig)', 'SG_Smoothed']], on='Pressure(Psig)', how='left')
        df_3= pd.merge(df_2, df_Density[['Pressure(Psig)', 'Corrected_Bod']], on='Pressure(Psig)', how='left')
        df_3['Bodb_new'] = df_3['Corrected_Bod']/df_3['Vrel_Smoothed']
        df_4 = df_3[df_3['Pressure(Psig)']<new_pb]
        df_8 = df_3[df_3['Pressure(Psig)'] < old_pb]
        df_6 = df_3[df_3['Pressure(Psig)']>new_pb]
        df_7 = df_3[df_3['Pressure(Psig)']==new_pb]
        Bodb_new_corrected=df_8['Bodb_new'].mean()
        df_4['Bod_Density_Corrected_Extended']=Bodb_new_corrected*df_4['Vrel_Smoothed']
        df_6['Bod_Density_Corrected_Extended'] = (Bodb_new_corrected) * (1 - (new_pb - df_6['Pressure(Psig)']) * (slope_CME))
        df_7['Bod_Density_Corrected_Extended']=Bodb_new_corrected
        Final_Density = pd.concat([df_6, df_7, df_4], ignore_index=True)
        Final_Density['Rsd_Diff'] = np.abs(Final_Density['Rsd_Smoothed'].diff())
        Final_Density['Gas_Volume'] = Final_Density['Rsd_Diff'].fillna(0)
        Final_Density['Gas_Mass'] = Final_Density['Gas_Volume'] * Final_Density['SG_Smoothed'] * 1.225
        Final_Density['Gas_Mass'] =Final_Density['Gas_Mass'].fillna(0)
        Final_Density['Oil+Gas_Mass'] = float('nan')
        Final_Density.loc[df.index[-1], 'Oil+Gas_Mass'] = 1000 * SG
        for i in range(len(Final_Density) - 2, -1, -1):
            Final_Density.loc[i, 'Oil+Gas_Mass'] = Final_Density.loc[i + 1, 'Oil+Gas_Mass'] + Final_Density.loc[i + 1, 'Gas_Mass']

        Final_Density['Oil_Density_Calculated'] = Final_Density['Oil+Gas_Mass'] / Final_Density['Bod_Density_Corrected_Extended'] / 1000
        Final_Density=Final_Density.rename(columns = {'Rsd_Smoothed': 'Rsd_Extended', 'SG_Smoothed':'SG_Extended'})
        Final_Density=Final_Density[['Pressure(Psig)', 'Pressure(Psia)', 'Bod_Density_Corrected_Extended', 'Rsd_Extended', 'SG_Extended', 'Oil_Density_Calculated']]
        y1 = Final_Density['Oil_Density_Calculated'][~Final_Density['Oil_Density_Calculated'].isna()].values
        x1 = Final_Density['Pressure(Psig)'][~Final_Density['Pressure(Psig)'].isna()].values
        matching_row = Final_Density.loc[Final_Density['Pressure(Psig)'] == new_pb, 'Bod_Density_Corrected_Extended']
        Bodb_new_S = matching_row.iloc[0]
        fig_density = {
            'data': [
                go.Scatter(x=x1, y=y1, mode='lines+markers', marker=dict(color='red', size=10),
                           name='Oil_Density_Calculated')],
            'layout': go.Layout(title='Oil Density at New Bubble Point Pressure', xaxis={'title': 'Pressure'},
                                yaxis={'title': 'Oil_Density'},
                                )}
        return Final_Density.to_dict('records'),fig_density,Bodb_new_S

    elif button_id == 'clear-button_Extended':
        return [{'Pressure(Psig)': '', 'Pressure(Psia)': '', 'Bod_Density_Corrected_Extended': '', 'Rsd_Extended': '', 'SG_Extended': '',
                 'Oil_Density_Calculated': ''} for _ in range(10)],go.Figure(),[]

    return current_data, go.Figure(),Bodb_new_S  # Default: return the current table data


# Separator corrections

@app.callback(
    Output('output-table_separator', 'columns'),
    Output('output-table_separator', 'data'),
    Output('extra-set-input', 'value'),
    Input('update-button', 'n_clicks'),
    State('extra-set-input', 'value'),
    State('output-table_separator', 'data')
)
def update_table_columns(n_clicks, extra_sets, table_data):
    # Start with the fixed columns
    columns = datatable_columns_separtor.copy()

    # Find the index after 'Separator_SG-1' to insert new columns
    insert_index = next(i for i, col in enumerate(columns) if col['id'] == 'Separator_SG-1') + 1

    # Add additional sets of columns based on user input
    if extra_sets and extra_sets > 0:
        for i in range(2, extra_sets + 1):  # Start from 2 for naming
            new_columns = [
                {'name': f'Separtor_Pressure-{i}', 'id': f'Separtor_Pressure-{i}'},
                {'name': f'Separtor_Temperature-{i}', 'id': f'Separtor_Temperature-{i}'},
                {'name': f'Separator_GOR-{i}', 'id': f'Separator_GOR-{i}'},
                {'name': f'Separator_SG-{i}', 'id': f'Separator_SG-{i}'}
            ]
            # Insert new columns after 'Separator_SG-1' and following each additional set
            columns[insert_index:insert_index] = new_columns
            insert_index += len(new_columns)  # Move insert index forward by the number of new columns

    # Update data to match the new columns
    updated_data = [{col['id']: row.get(col['id'], '') for col in columns} for row in table_data]

    return columns, updated_data,extra_sets

@app.callback(
    Output('Bodb_old-input_separator', 'value'),
     Output('Bodb_new-input_separator', 'value'),
     Output('Rsdb_old-input_separator', 'value'),
     Output('Rsdb_new-input_separator', 'value'),
     Output('density-input_separator', 'value'),
    Input('use-previous-checkbox', 'value'),
    State('Rsdb_old_S','data'),
    State('Rsdb_new_S','data'),
    State('Bodb_old_S','data'),
    State('Bodb_new_S','data'),
    State('Density_S','data'),
    prevent_initial_call=True
)
def fill_inputs_based_on_checkbox(use_previous,Rsdb_old,Rsdb_new,Bodb_old,Bodb_new,Density_oil_old):
    if 'use_previous' in use_previous:
        Rsdb_old = Rsdb_old
        Bodb_old= Bodb_old
        Rsdb_new=Rsdb_new
        Bodb_new=Bodb_new
        Density_oil_old=Density_oil_old
        return Bodb_old, Bodb_new,Rsdb_old, Rsdb_new,  Density_oil_old  # Example previous values
    return None, None, None, None, None  # Clear inputs if checkbox is unchecked


@app.callback(

Output('output-table_separator', 'data',allow_duplicate=True),
    Input('submit-button_separator', 'n_clicks'),
    Input('clear-button_separator', 'n_clicks'),
    State('Bodb_old-input_separator', 'value'),
     State('Bodb_new-input_separator', 'value'),
     State('Rsdb_old-input_separator', 'value'),
     State('Rsdb_new-input_separator', 'value'),
     State('density-input_separator', 'value'),
    State('extra-set-input', 'value'),
    State('output-table_separator', 'data'),
    State('output-table_separator', 'columns'),
    prevent_initial_call=True
)
def update_table(submit_clicks,clear_clicks,Bodb_old,Bodb_new,Rsdb_old,Rsdb_new,Density_oil_old,S_stage,current_data,columns):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_data


    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'submit-button_separator':
        df_table = pd.DataFrame(current_data)
        numeric_columns = ['Separtor_Pressure-1', 'Separtor_Temperature-1', 'Separator_GOR-1', 'Separator_SG-1',
                           'Stock_Tank_GOR', 'Stock_Tank_SG', 'Stock_Tank_Oil_Gravity', 'Bofb_Old_Lab',
                           'Density_at_Old_Pb', 'Bofb_Old_Density_Corrected', 'Bofb_New', 'Rsfb_New']
        df_table = df_table[numeric_columns].apply(pd.to_numeric, errors='coerce')
        if S_stage==1:
            df_table['Density_at_Old_Pb'] = (df_table['Stock_Tank_Oil_Gravity']+(0.001225*(df_table['Separator_GOR-1']*df_table['Separator_SG-1']+df_table['Stock_Tank_GOR']*df_table['Stock_Tank_SG'])))/df_table['Bofb_Old_Lab']
            df_table['Bofb_Old_Density_Corrected'] = (df_table['Stock_Tank_Oil_Gravity']+(0.001225*df_table['Separator_GOR-1']*df_table['Separator_SG-1']))/(Density_oil_old)
            df_table['Bofb_New'] = (Bodb_new/Bodb_old)*df_table['Bofb_Old_Density_Corrected']
            df_table['Rsfb_New'] = (Rsdb_new / Rsdb_old) * (df_table['Stock_Tank_GOR']+df_table['Separator_GOR-1'])

        return df_table.to_dict('records')

    if button_id == 'clear-button_separator':
        columns = columns.copy()

        cleared_data = [
        {col['id']: (row[col['id']] if col['id'] == 'Separator_Test' else '') for col in columns}
        for row in current_data
    ]
        return cleared_data
    return current_data







# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)



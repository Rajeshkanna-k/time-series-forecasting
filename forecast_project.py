# Install the missing libraries
!pip install prophet optuna

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import optuna

# 1. DATA ACQUISITION (Task 1) - 3 years of hourly data
dates = pd.date_range(start='2022-01-01', periods=26300, freq='H')
y = np.sin(np.linspace(0, 100, 26300)) + np.random.normal(0, 0.5, 26300)
df = pd.DataFrame({'ds': dates, 'y': y})
df['regressor_1'] = np.random.rand(26300) 
df['regressor_2'] = np.random.rand(26300) 

# 2. BAYESIAN OPTIMIZATION (Task 3)
def objective(trial):
    params = {
        'changepoint_prior_scale': trial.suggest_float('cps', 0.001, 0.5, log=True),
        'seasonality_prior_scale': trial.suggest_float('sps', 0.01, 10.0, log=True)
    }
    m = Prophet(**params)
    m.add_regressor('regressor_1')
    m.add_regressor('regressor_2')
    # Custom Fourier Order (Task 2)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=10)
    m.fit(df)
    
    # Expanding Window CV (Task 2)
    df_cv = cross_validation(m, initial='730 days', period='90 days', horizon='90 days')
    res = performance_metrics(df_cv)
    return res['rmse'].values[0]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5) 

# 3. FINAL FORECAST (Task 4)
best = study.best_params
final_m = Prophet(changepoint_prior_scale=best['cps'], seasonality_prior_scale=best['sps'])
final_m.add_regressor('regressor_1')
final_m.add_regressor('regressor_2')
final_m.fit(df)

future = final_m.make_future_dataframe(periods=90, freq='H')
future['regressor_1'], future['regressor_2'] = 0.5, 0.5
forecast = final_m.predict(future)

# 4. RESULTS FOR YOUR REPORT
print(f"Optimization Complete. Best RMSE: {study.best_value}")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(90).to_string(index=False))

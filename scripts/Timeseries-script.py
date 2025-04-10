import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('average-monthly-surface-temperature.csv')

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

# Data preprocessing
print("\nPerforming data preprocessing...")

# Convert Day to datetime
data['Day'] = pd.to_datetime(data['Day'])

# Check for missing values
print(f"\nMissing values:\n{data.isnull().sum()}")

# Select a specific country for time series analysis (e.g., global average or a specific country)
# For this example, let's use global average data
country = 'World'
print(f"\nPerforming time series analysis for: {country}")

# Filter data for the selected country
country_data = data[data['Entity'] == country].copy()

# If no 'World' entity exists, use a large country with good data quality
if len(country_data) == 0:
    print(f"No data found for {country}, using United States instead")
    country = 'United States'
    country_data = data[data['Entity'] == country].copy()

# Sort data by date
country_data = country_data.sort_values('Day')

# Set Day as index
country_data.set_index('Day', inplace=True)

# Visualize the time series
plt.figure(figsize=(12, 6))
plt.plot(country_data.index, country_data['Average surface temperature'])
plt.title(f'Average Surface Temperature for {country} Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.savefig(f'{country}_temperature_time_series.png')
plt.close()

# Check for stationarity using Augmented Dickey-Fuller test
print("\nChecking for stationarity...")
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries.dropna())
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    # Interpret the results
    if result[1] <= 0.05:
        print("Stationary (reject the null hypothesis)")
        return True
    else:
        print("Non-stationary (fail to reject the null hypothesis)")
        return False

# Check stationarity of the original series
ts = country_data['Average surface temperature']
original_stationary = check_stationarity(ts)

# If the series is not stationary, apply differencing
if not original_stationary:
    print("\nApplying differencing to make the series stationary...")
    ts_diff = ts.diff().dropna()
    
    plt.figure(figsize=(12, 6))
    plt.plot(ts_diff)
    plt.title(f'Differenced Temperature Series for {country}')
    plt.xlabel('Date')
    plt.ylabel('Temperature Change (°C)')
    plt.grid(True)
    plt.savefig(f'{country}_differenced_series.png')
    plt.close()
    
    # Check stationarity of the differenced series
    diff_stationary = check_stationarity(ts_diff)
    
    # Use the differenced series if it's stationary
    if diff_stationary:
        ts_model = ts_diff
        d_order = 1
    else:
        # Try second-order differencing if needed
        ts_diff2 = ts_diff.diff().dropna()
        diff2_stationary = check_stationarity(ts_diff2)
        if diff2_stationary:
            ts_model = ts_diff2
            d_order = 2
        else:
            # If still not stationary, proceed with caution
            print("Warning: Series remains non-stationary after differencing")
            ts_model = ts_diff
            d_order = 1
else:
    ts_model = ts
    d_order = 0

# Plot ACF and PACF to determine AR and MA orders
plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(ts_model.dropna(), ax=plt.gca(), lags=40)
plt.subplot(212)
plot_pacf(ts_model.dropna(), ax=plt.gca(), lags=40)
plt.tight_layout()
plt.savefig(f'{country}_acf_pacf.png')
plt.close()

# Feature engineering for time series
print("\nPerforming feature engineering for time series...")

# Create a DataFrame with the original series
ts_features = pd.DataFrame(ts)

# Add lag features
for lag in range(1, 13):  # 12 months of lag features
    ts_features[f'lag_{lag}'] = ts.shift(lag)

# Add rolling statistics
for window in [3, 6, 12]:  # 3, 6, and 12-month windows
    ts_features[f'rolling_mean_{window}'] = ts.rolling(window=window).mean()
    ts_features[f'rolling_std_{window}'] = ts.rolling(window=window).std()

# Add seasonal features
ts_features['month'] = ts_features.index.month
ts_features['quarter'] = ts_features.index.quarter
ts_features['year'] = ts_features.index.year

# Create seasonal dummy variables
ts_features = pd.get_dummies(ts_features, columns=['month', 'quarter'], drop_first=True)

# Drop NaN values created by lag and rolling features
ts_features = ts_features.dropna()

print("\nFeature engineering complete. Features created:")
print(ts_features.columns.tolist())

# Time series cross-validation
print("\nPerforming time series cross-validation...")

# Prepare data for modeling
X = ts_features.drop('Average surface temperature', axis=1)
y = ts_features['Average surface temperature']

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# SARIMA modeling
print("\nFitting SARIMA model...")

# Determine optimal p, d, q, P, D, Q, s parameters based on ACF/PACF plots
# For simplicity, we'll use some common values, but in practice, you'd use grid search
p, d, q = 1, d_order, 1  # Non-seasonal components
P, D, Q, s = 1, 1, 1, 12  # Seasonal components (s=12 for monthly data)

# Create and fit SARIMA model
try:
    model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s),
                   enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    print("\nSARIMA Model Summary:")
    print(results.summary().tables[1])
    
    # Forecast
    forecast_steps = 24  # 2 years forecast
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='MS')[1:]
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
    
    # Confidence intervals
    conf_int = forecast.conf_int()
    
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts, label='Historical Data')
    plt.plot(forecast_series.index, forecast_series, color='red', label='Forecast')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.title(f'SARIMA Forecast for {country} Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{country}_sarima_forecast.png')
    plt.close()
    
    # In-sample predictions
    in_sample_pred = results.get_prediction(start=ts.index[0], end=ts.index[-1])
    in_sample_mean = in_sample_pred.predicted_mean
    
    # Calculate metrics
    mse = mean_squared_error(ts[in_sample_mean.index], in_sample_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ts[in_sample_mean.index], in_sample_mean)
    r2 = r2_score(ts[in_sample_mean.index], in_sample_mean)
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot residuals
    residuals = ts[in_sample_mean.index] - in_sample_mean
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(residuals)
    plt.title('Residuals')
    plt.grid(True)
    
    plt.subplot(212)
    plt.hist(residuals, bins=30)
    plt.title('Residuals Distribution')
    plt.tight_layout()
    plt.savefig(f'{country}_sarima_residuals.png')
    plt.close()
    
    # Alternative model: ARIMA
    print("\nFitting ARIMA model for comparison...")
    arima_model = ARIMA(ts, order=(p, d, q))
    arima_results = arima_model.fit()
    
    # ARIMA forecast
    arima_forecast = arima_results.forecast(steps=forecast_steps)
    arima_forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='MS')[1:]
    arima_forecast_series = pd.Series(arima_forecast, index=arima_forecast_index)
    
    # Plot ARIMA forecast
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts, label='Historical Data')
    plt.plot(arima_forecast_series.index, arima_forecast_series, color='green', label='ARIMA Forecast')
    plt.title(f'ARIMA Forecast for {country} Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{country}_arima_forecast.png')
    plt.close()
    
    # Compare SARIMA and ARIMA
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts, label='Historical Data', alpha=0.7)
    plt.plot(forecast_series.index, forecast_series, color='red', label='SARIMA Forecast')
    plt.plot(arima_forecast_series.index, arima_forecast_series, color='green', label='ARIMA Forecast')
    plt.title(f'SARIMA vs ARIMA Forecast for {country} Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{country}_sarima_vs_arima.png')
    plt.close()
    
    print("\nTime series analysis and forecasting completed successfully!")
    print(f"Forecast images saved as {country}_sarima_forecast.png and {country}_arima_forecast.png")
    
except Exception as e:
    print(f"Error in SARIMA modeling: {e}")
    print("Trying simpler ARIMA model instead...")
    
    # Fallback to ARIMA model
    try:
        arima_model = ARIMA(ts, order=(p, d, q))
        arima_results = arima_model.fit()
        
        print("\nARIMA Model Summary:")
        print(arima_results.summary().tables[1])
        
        # ARIMA forecast
        forecast_steps = 24  # 2 years forecast
        arima_forecast = arima_results.forecast(steps=forecast_steps)
        arima_forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='MS')[1:]
        arima_forecast_series = pd.Series(arima_forecast, index=arima_forecast_index)
        
        # Plot ARIMA forecast
        plt.figure(figsize=(12, 6))
        plt.plot(ts.index, ts, label='Historical Data')
        plt.plot(arima_forecast_series.index, arima_forecast_series, color='green', label='ARIMA Forecast')
        plt.title(f'ARIMA Forecast for {country} Temperature')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{country}_arima_forecast.png')
        plt.close()
        
        # In-sample predictions for ARIMA
        in_sample_pred = arima_results.get_prediction(start=ts.index[0], end=ts.index[-1])
        in_sample_mean = in_sample_pred.predicted_mean
        
        # Calculate metrics for ARIMA
        mse = mean_squared_error(ts[in_sample_mean.index], in_sample_mean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(ts[in_sample_mean.index], in_sample_mean)
        r2 = r2_score(ts[in_sample_mean.index], in_sample_mean)
        
        print("\nARIMA Model Evaluation:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        print("\nTime series analysis and forecasting completed with ARIMA model!")
        print(f"Forecast image saved as {country}_arima_forecast.png")
        
    except Exception as e:
        print(f"Error in ARIMA modeling: {e}")
        print("Consider using a simpler model or different parameters.")

print("\nTime series analysis script execution completed.")
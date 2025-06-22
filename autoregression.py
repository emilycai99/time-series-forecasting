import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, acf, q_stat
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
import matplotlib.pyplot as plt


def check_stationarity(series, threshold=0.05):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    return result[1] > threshold  # Returns True if non-stationary

def make_data_stationary(df):
    diff_flag = False
    # Check each series
    for col in df.columns:
        print(f"Checking {col}:")
        if check_stationarity(df[col]):
            print(f"{col} is non-stationary - differencing required")
            diff_flag = True
        else:
            print(f"{col} is stationary")

    if diff_flag:
        # First difference the non-stationary series
        df = df.diff().dropna()

        # Verify stationarity after differencing
        for col in df.columns:
            print(f"Checking differenced {col}, stationary: {not check_stationarity(df[col])}")

    return df

def fit_VAR(df, lag=None):
    # Create the VAR model
    model = VAR(df)

    if lag is None:
        # Select optimal lag order using information criteria
        lag_results = model.select_order(maxlags=10)  # Try up to 10 lags
        print(lag_results.summary())

        # Choose the lag with the lowest AIC/BIC/HQIC (depends on your preference)
        optimal_lag = lag_results.aic  # or .bic, .hqic

        # Fit the model with the selected lag order
        var_model = model.fit(optimal_lag)
    else:
        var_model = model.fit(lag)

    # View model summary
    print(var_model.summary())
                
    return var_model

def diagnostics_checking(results):
    ## Get residuals
    residuals = results.resid

    ## Plot ACF and PACF for each residual series
    for i, col in enumerate(residuals.columns):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        plot_acf(residuals[col], lags=20, alpha=0.05, title=f'ACF of Residuals - {col}', ax=axes[0])
        
        # QQ plot
        qqplot(residuals[col], line='q', fit=True, ax=axes[1])
        axes[1].set_title(f'QQ Plot - {col}')
        
        plt.tight_layout()
        plt.show()

    ## Ljung-Box Test for autocorrelation
    print("\nLjung-Box Test Results:")
    for col in residuals.columns:
        lb_test = q_stat(acf(residuals[col], nlags=10), len(residuals[col]))[1]
        print(f"\n{col}:")
        for lag, p_val in enumerate(lb_test, 1):
            print(f"Lag {lag}: p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else '(not significant)'}")

    ## 6. Normality tests
    print("\nNormality Tests (Shapiro-Wilk):")
    for col in residuals.columns:
        sw_test = stats.shapiro(residuals[col])
        print(f"{col}: p-value = {sw_test[1]:.4f} {'(non-normal)' if sw_test[1] < 0.05 else '(normal)'}")

# Revert differencing
def invert_transformation(data, diff_data, window_size, scaler):
    """
    data: dataframe before differencing and scaling
    diff_data: dataframe after differencing and scaling
    """
    original = data.copy()
    columns = data.columns
    target_idx = data.iloc[[window_size-1]].index
    # revert the standardization
    diff_data = scaler.inverse_transform(diff_data)
    for col in columns:
        if col != 'date':
            # TODO: consider the standardization
            forecast_df = diff_data[col].cumsum() + data[col].iloc[window_size-1]
            forecast_df.index = original.index[window_size:]
            original[str(col)+'_forecast'] = forecast_df
            original.loc[target_idx, str(col)+'_forecast'] = data.loc[target_idx, str(col)]
    return original.dropna()


def forecast(model, original_df, diff_df, scaler, steps=1):
    # rolling window approach
    # Iterate over the testing dataset using a rolling window
    window_size = model.k_ar
    print('window_size', window_size)
    # Initialize an empty DataFrame to store the predictions
    predictions = pd.DataFrame(columns=diff_df.columns)

    # make sure everything has the corresponding true value
    for i in range(len(diff_df) - window_size + 1 - steps):
        test_window = diff_df.iloc[i:i + window_size]
        forecast = model.forecast(test_window.values, steps=steps)
        forecast_df = pd.DataFrame(forecast, columns=diff_df.columns)
        predictions = pd.concat([predictions, forecast_df], ignore_index=True)

    forecast_values = invert_transformation(original_df, predictions, window_size, scaler)

    return forecast_values, predictions

def forecast_metric(df, target_features):
    store_mae = []
    store_mse = []
    for col in target_features:
        true = df[col]
        pred = df[str(col)+'_forecast']
        store_mae.append(np.abs(true - pred))
        store_mse.append(np.square(true - pred))
    
    mae = np.mean(np.concatenate(store_mae))
    mse = np.mean(np.concatenate(store_mse))

    print(f'MAE: {round(mae, 4)}, MSE: {round(mse, 4)}')
    
    return mae, mse

def plot_forecast(df, target_features):
    fig, axes = plt.subplots(len(target_features), 1, figsize=(12, 6))

    for i, col in enumerate(target_features):
        axes[i].plot(df.index, df[col], label='True Values', marker='o')
        axes[i].plot(df.index, df[str(col) + '_forecast'], label='Forecasted Values', marker='x')

        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'{col}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from data_loader import ts_dataset
    
    # Simulate data
    total_size = 100
    # Generate time index
    dates = pd.date_range('2020-01-01', periods=total_size)

    # Linear trend + noise
    linear_trend1 = 0.1 * np.arange(total_size) + np.random.normal(0, 2, total_size)
    linear_trend2 = 0.2 * np.arange(total_size) + np.random.normal(0, 1, total_size)
    linear_trend3 = 0.3 * np.arange(total_size) + np.random.normal(0, 2, total_size)

    non_stationary_df = pd.DataFrame(np.concatenate([np.expand_dims(linear_trend1, 1), np.expand_dims(linear_trend2, 1), 
            np.expand_dims(linear_trend3, 1)], axis=1), 
            index=dates, columns=['a', 'b', 'c'])
    
    df = make_data_stationary(non_stationary_df)

    # note that in train_dataset, seq_len / pred_len does not matter
    train_dataset = ts_dataset(df, 2, 1, 'train', ['a', 'b', 'c'], scale=True, inverse=True)
    train_df = train_dataset.get_dataframe()
    scaler = train_dataset.get_scaler()

    # fit model
    model = fit_VAR(train_df, 2)
    # check residuals
    diagnostics_checking(model)

    # note that for autoregression, we do not use the validation
    # hence pool the two together
    # seq_len depends on model.k_ar
    val_dataset = ts_dataset(df, int(model.k_ar), 1, 'val', ['a', 'b', 'c'], scale=True, inverse=True)
    val_df = val_dataset.get_dataframe()

    test_dataset = ts_dataset(df, int(model.k_ar), 1, 'test', ['a', 'b', 'c'], scale=True, inverse=True)
    test_df = test_dataset.get_dataframe()

    # important to drop_duplicates
    test_df = pd.concat([val_df, test_df]).drop_duplicates()

    non_stationary_test_df = non_stationary_df.loc[non_stationary_df.index.isin(test_df.index)]

    forecast_df, forecast_diff_df = forecast(model, non_stationary_test_df, test_df, scaler, 1)

    forecast_metric(forecast_df, target_features=['a', 'b', 'c'])

    plot_forecast(forecast_df, target_features=['a', 'b', 'c'])
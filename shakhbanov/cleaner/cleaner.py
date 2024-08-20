import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from typing import Union
from joblib import Parallel, delayed

class DataCleaner:
    def __init__(self, method: str = 'mean', weeks: int = 4, window: int = 365, std: int = 2, tqdm: bool = False, plot: bool = False, num_plot: int = 5, parallel: bool = False):
        """
        Initialize the DataCleaner class with specified parameters.
        
        :param method: Method to calculate central tendency ('mean', 'median', or 'mode')
        :param weeks: Number of weeks to consider for rolling mean calculation
        :param window: Window size for moving average and standard deviation
        :param std: Number of standard deviations to determine outliers
        :param tqdm: Flag to use tqdm progress bar
        :param plot: Flag to enable or disable visualization
        :param num_plot: Number of objects to visualize
        :param parallel: Flag to use parallel processing
        """
        assert method in ['mean', 'median', 'mode'], "Method must be 'mean', 'median', or 'mode'"
        assert weeks > 0, "Weeks must be greater than 0"
        assert window > 0, "Window size must be greater than 0"
        assert std > 0, "Standard deviation multiplier must be greater than 0"

        self.method = method
        self.weeks = weeks
        self.window = window
        self.std = std
        self.tqdm = tqdm
        self.plot = plot
        self.num_plot = num_plot
        self.parallel = parallel

    def calculate_central_tendency(self, data: pd.Series) -> Union[float, int]:
        """
        Calculate the central tendency based on the specified method.
        
        :param data: Data to calculate central tendency on
        :return: Central tendency value
        """
        if self.method == 'mean':
            return data.mean()
        elif self.method == 'median':
            return data.median()
        elif self.method == 'mode':
            return data.mode().iloc[0]

    def calculate_week_average(self, day_data: pd.DataFrame, column: str, reference_date: pd.Timestamp, lower_bound: float, upper_bound: float) -> float:
        """
        Calculate the central tendency for the past 4 weeks.
        
        :param day_data: DataFrame with day data
        :param column: Column to calculate central tendency on
        :param reference_date: Reference date to calculate from
        :param lower_bound: Lower bound for outliers
        :param upper_bound: Upper bound for outliers
        :return: Central tendency value for the past 4 weeks
        """
        start_date = reference_date - pd.DateOffset(weeks=self.weeks)
        relevant_data = day_data[(day_data['ds'] < reference_date) & (day_data['ds'] >= start_date)]
        relevant_data = relevant_data[(relevant_data[column] >= lower_bound) & (relevant_data[column] <= upper_bound)]
        if relevant_data.empty:
            return self.calculate_central_tendency(day_data[column])  # In case of no data for the last 4 weeks, use overall central tendency
        return self.calculate_central_tendency(relevant_data[column])

    def process_series(self, series_data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Process a single time series to clean data.
        
        :param series_data: DataFrame with series data
        :param columns: List of columns to clean
        :return: Cleaned series DataFrame
        """
        series_data = series_data.copy()
        for column in columns:
            # Calculate moving average and standard deviation
            series_data[f'moving_average_{column}'] = series_data[column].rolling(window=self.window, min_periods=1, center=True).mean()
            series_data[f'std_dev_{column}'] = series_data[column].rolling(window=self.window, min_periods=1, center=True).std()
            series_data[f'lower_{column}'] = series_data[f'moving_average_{column}'] - self.std * series_data[f'std_dev_{column}']
            series_data[f'upper_{column}'] = series_data[f'moving_average_{column}'] + self.std * series_data[f'std_dev_{column}']

            # Replace outliers with the mean value for the given day of the week over the last 4 weeks
            for day in range(7):
                day_data = series_data[series_data['weekday'] == day]
                for idx in day_data.index:
                    if day_data.loc[idx, column] < day_data.loc[idx, f'lower_{column}'] or day_data.loc[idx, column] > day_data.loc[idx, f'upper_{column}']:
                        reference_date = day_data.loc[idx, 'ds']
                        mean_value = self.calculate_week_average(day_data, column, reference_date, day_data.loc[idx, f'lower_{column}'], day_data.loc[idx, f'upper_{column}'])
                        series_data.loc[idx, column] = mean_value

        return series_data

    def clean_data(self, data: pd.DataFrame, columns: list, series_id: str = 'series_id') -> pd.DataFrame:
        """
        Clean the data by replacing outliers with central tendency values.
        
        :param data: DataFrame with data to clean
        :param columns: List of columns to clean
        :param series_id: Column name that identifies different time series
        :return: Cleaned DataFrame
        """
        assert 'ds' in data.columns, "Column 'ds' not found in DataFrame"
        assert all(col in data.columns for col in columns), "One or more columns not found in DataFrame"
        assert series_id in data.columns, f"Column {series_id} not found in DataFrame"

        data = data.copy()
        if not np.issubdtype(data['ds'].dtype, np.datetime64):
            data['ds'] = pd.to_datetime(data['ds'])

        # Adding 'weekday' column
        data.loc[:, 'weekday'] = data['ds'].dt.weekday

        restored_data = data.copy()
        unique_series_ids = restored_data[series_id].unique()

        if self.tqdm:
            unique_series_ids = tqdm(unique_series_ids)

        if self.parallel:
            results = Parallel(n_jobs=-1)(delayed(self.process_series)(restored_data[restored_data[series_id] == series], columns) for series in unique_series_ids)
            restored_data = pd.concat(results)
        else:
            restored_data = pd.concat([self.process_series(restored_data[restored_data[series_id] == series], columns) for series in unique_series_ids])

        # Removing 'weekday' column
        restored_data.drop(columns=['weekday'], inplace=True)

        # Removing temporary columns
        for column in columns:
            restored_data.drop(columns=[f'moving_average_{column}', f'std_dev_{column}', f'lower_{column}', f'upper_{column}'], inplace=True)

        # Visualization
        if self.plot:
            self.plot_data(data, restored_data, columns, series_id)

        return restored_data

    def plot_data(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame, columns: list, series_id: str):
        """
        Visualize the original and cleaned data for the specified number of objects.
        
        :param original_data: Original DataFrame before cleaning
        :param cleaned_data: DataFrame after cleaning
        :param columns: List of columns being cleaned
        :param series_id: Column name that identifies different time series
        """
        unique_series_ids = original_data[series_id].unique()[:self.num_plot]

        for series in unique_series_ids:
            original_series_data = original_data[original_data[series_id] == series]
            cleaned_series_data = cleaned_data[cleaned_data[series_id] == series]

            for column in columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(x=original_series_data['ds'], y=original_series_data[column], c='r', alpha=.5, marker='x', s=30, label='Original Data')
                plt.scatter(x=cleaned_series_data['ds'], y=cleaned_series_data[column], c='#0072b2', label='Cleaned Data')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{series_id.capitalize()} {series} - {column}')
                plt.legend()
                plt.show()
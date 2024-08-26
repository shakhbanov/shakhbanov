import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from statsmodels.tsa.seasonal import seasonal_decompose

class NaturalGapFiller:
    def __init__(self, date_column: str, group_column: str, target_columns: list[str], tqdm: bool = True, parallel: bool = True, max_deviation: float = 3.0, seasonality_period: int = 30):
        """
        Initialize the NaturalGapFiller.

        Parameters:
        - date_column (str): The name of the date column.
        - group_column (str): The name of the group column.
        - target_columns (list[str]): List of target column names to apply the filling.
        - tqdm (bool): Whether to use tqdm for progress visualization.
        - parallel (bool): Whether to use parallel processing.
        - max_deviation (float): The maximum deviation for added noise.
        - seasonality_period (int): The period of seasonality for time series decomposition.
        """
        assert isinstance(date_column, str), "date_column should be a string"
        assert isinstance(group_column, str), "group_column should be a string"
        assert isinstance(target_columns, list) and all(isinstance(col, str) for col in target_columns), "target_columns should be a list of strings"
        assert isinstance(tqdm, bool), "tqdm should be a boolean"
        assert isinstance(parallel, bool), "parallel should be a boolean"
        assert isinstance(max_deviation, float) and max_deviation > 0, "max_deviation should be a positive float"
        assert isinstance(seasonality_period, int) and seasonality_period > 0, "seasonality_period should be a positive integer"

        self.date_column = date_column
        self.group_column = group_column
        self.target_columns = target_columns
        self.tqdm = tqdm
        self.parallel = parallel
        self.max_deviation = max_deviation
        self.seasonality_period = seasonality_period

    def _create_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a full date range for the given DataFrame, filling in any missing dates.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with a full date range.
        """
        assert isinstance(df, pd.DataFrame), "df should be a pandas DataFrame"
        assert self.date_column in df.columns, f"{self.date_column} is not in the DataFrame columns"

        full_range = pd.date_range(start=df[self.date_column].min(), end=df[self.date_column].max())
        group_value = df[self.group_column].iloc[0]
        df_full = df.set_index(self.date_column).reindex(full_range).rename_axis(self.date_column).reset_index()
        df_full[self.group_column] = group_value
        return df_full

    def _find_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify gaps in the date column of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with an additional column 'gaps' that indicates gaps in the date sequence.
        """
        assert isinstance(df, pd.DataFrame), "df should be a pandas DataFrame"
        assert self.date_column in df.columns, f"{self.date_column} is not in the DataFrame columns"

        df = df.sort_values(by=self.date_column)
        df['next_day'] = df[self.date_column].shift(-1)
        df['gap'] = (df['next_day'] - df[self.date_column]).dt.days - 1
        
        gaps = []
        for index, row in df.iterrows():
            if row['gap'] > 0:
                gaps.append(f"{row[self.date_column].date()} - {row['next_day'].date()}")
            else:
                gaps.append(None)
        
        df['gaps'] = gaps
        df.drop(columns=['next_day', 'gap'], inplace=True)
        return df

    def _apply_randomized_interpolation(self, series: pd.Series, trend: pd.Series) -> pd.Series:
        """
        Apply randomized interpolation to fill missing values, taking trend into account.

        Parameters:
        - series (pd.Series): The time series with missing values.
        - trend (pd.Series): The trend component of the time series.

        Returns:
        - pd.Series: The series with missing values filled and trend added back.
        """
        assert isinstance(series, pd.Series), "series should be a pandas Series"
        assert isinstance(trend, pd.Series), "trend should be a pandas Series"
        assert len(series) == len(trend), "series and trend should have the same length"

        detrended_series = series - trend  # Remove trend before interpolation
        observed_values = detrended_series.copy()
        mask = np.isnan(observed_values)

        if np.sum(mask) == 0:
            return observed_values + trend  # Add trend back after processing

        # Use piecewise linear interpolation for initial filling
        valid_idx = np.where(~mask)[0]
        valid_values = observed_values[~mask]
        assert len(valid_idx) > 1, "Not enough valid data points to perform interpolation"

        f = interp1d(valid_idx, valid_values, kind='linear', fill_value="extrapolate")
        interpolated_values = f(range(len(observed_values)))

        # Add limited random noise based on the amplitude of neighboring values
        for i in np.where(mask)[0]:
            local_std = np.std(valid_values[max(0, i-5):min(len(valid_values), i+5)])
            deviation = np.random.normal(0, local_std)
            deviation = np.clip(deviation, -self.max_deviation * local_std, self.max_deviation * local_std)  # Limit deviation
            interpolated_values[i] += deviation

        # Apply median filter to smooth out any potential outliers
        smoothed_values = median_filter(interpolated_values, size=3)

        observed_values[mask] = smoothed_values[mask]
        return observed_values + trend  # Add trend back after processing

    def _process_group(self, group_value: int, group: pd.DataFrame) -> pd.DataFrame:
        """
        Process each group by filling in missing values considering trends and seasonality.

        Parameters:
        - group_value (int): The value identifying the group.
        - group (pd.DataFrame): The DataFrame corresponding to the group.

        Returns:
        - pd.DataFrame: The processed DataFrame with filled missing values.
        """
        assert isinstance(group_value, int), "group_value should be an integer"
        assert isinstance(group, pd.DataFrame), "group should be a pandas DataFrame"

        group = self._create_date_range(group)
        for column in self.target_columns:
            assert column in group.columns, f"Column '{column}' does not exist in the DataFrame."
            group[column] = group[column].replace(0, np.nan)  

            # Decompose the series to extract the trend component
            decomposition = seasonal_decompose(group[column].interpolate(method='linear'), model='additive', period=self.seasonality_period)
            trend = decomposition.trend.bfill().ffill()

            # Fill the data considering the trend
            group[column] = self._apply_randomized_interpolation(group[column], trend)
        group = self._find_gaps(group)
        return group

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the entire DataFrame to fill in missing values, considering trends and seasonality.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.DataFrame: The DataFrame with missing values filled.
        """
        assert isinstance(df, pd.DataFrame), "df should be a pandas DataFrame"
        assert self.date_column in df.columns, f"Column '{self.date_column}' does not exist in the DataFrame."
        assert self.group_column in df.columns, f"Column '{self.group_column}' does not exist in the DataFrame."

        if self.tqdm:
            iterable = tqdm(df.groupby(self.group_column), desc="Processing groups")
        else:
            iterable = df.groupby(self.group_column)
        
        if self.parallel:
            results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(self._process_group)(group_value, group) for group_value, group in iterable)
        else:
            results = [self._process_group(group_value, group) for group_value, group in iterable]
        
        df_filled = pd.concat(results).reset_index(drop=True)
        df_filled = df_filled.drop(columns=['gaps'])
        return df_filled

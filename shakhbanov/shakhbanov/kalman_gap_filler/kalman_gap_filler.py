import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from typing import List
from pykalman import KalmanFilter
from joblib import Parallel, delayed

class KalmanGapFiller:
    def __init__(self, date_column: str, group_column: str, target_columns: List[str], tqdm: bool = True, parallel: bool = True):
        """
        Initialize the KalmanGapFiller.

        Parameters:
        - date_column: The name of the column containing dates.
        - group_column: The name of the column containing group identifiers.
        - target_columns: A list of column names to which Kalman filter should be applied.
        - tqdm: Whether to use tqdm for progress visualization.
        - parallel: Whether to use parallel processing.
        """
        self.date_column = date_column
        self.group_column = group_column
        self.target_columns = target_columns
        self.tqdm = tqdm
        self.parallel = parallel
    
    def _create_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        full_range = pd.date_range(start=df[self.date_column].min(), end=df[self.date_column].max())
        group_value = df[self.group_column].iloc[0]
        df_full = df.set_index(self.date_column).reindex(full_range).rename_axis(self.date_column).reset_index()
        df_full[self.group_column] = group_value
        return df_full

    def _find_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _apply_kalman_filter_to_missing(self, series: pd.Series) -> pd.Series:
        observed_values = series.copy()
        mask = np.isnan(observed_values)

        if np.sum(mask) == 0:
            return observed_values

        temporary_filled = observed_values.copy()
        temporary_filled[mask] = 0

        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        state_means, _ = kf.em(temporary_filled, n_iter=10).smooth(temporary_filled)
        filtered_values = state_means.flatten()

        observed_values[mask] = filtered_values[mask]
        return observed_values

    def _process_group(self, group_value: int, group: pd.DataFrame) -> pd.DataFrame:
        group = self._create_date_range(group)
        for column in self.target_columns:
            assert column in group.columns, f"Column '{column}' does not exist in the dataframe."
            group[column] = group[column].replace(0, np.nan)  
            group[column] = self._apply_kalman_filter_to_missing(group[column].values)
        group = self._find_gaps(group)
        return group

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataframe to fill missing values and identify gaps.
        Parameters:
        - df: The input dataframe.
        Returns:
        - A processed dataframe with missing values filled and gaps identified.
        """
        assert self.date_column in df.columns, f"Column '{self.date_column}' does not exist in the dataframe."
        assert self.group_column in df.columns, f"Column '{self.group_column}' does not exist in the dataframe."
        
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

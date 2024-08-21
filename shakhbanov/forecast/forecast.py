import re, os, prophet
import numpy as np
import pandas as pd
import lightgbm as lgb

from prophet import Prophet
from holidays import Russia
from itertools import product
from tqdm.notebook import tqdm
from typing import Optional, Union
from sklearn.metrics import mean_squared_error
from holidays.holiday_base import HolidayBase
from sklearn.model_selection import TimeSeriesSplit

prophet.diagnostics.logging.disable(level=50)

class SuppressOutput:
    """
    Context manager for suppressing stdout and stderr in Python, including
    output from C/Fortran subfunctions. This is useful for silencing verbose
    output from backend computations.

    Exceptions are not suppressed, as they are printed to stderr before
    script termination.
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Restore the original stdout/stderr
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class LightGBMModel:
    """
    A class to encapsulate LightGBM model training and cross-validation with TimeSeriesSplit.

    Attributes:
        params (dict): Parameters for LightGBM model.
        n_splits (int): Number of splits for TimeSeriesSplit.
        early_stopping_rounds (int): Rounds of early stopping.
        n_estimators (int): Number of boosting iterations.
        best_iteration (int): Best iteration determined by cross-validation.
        best_score (float): Best score achieved during cross-validation.
        model (lgb.LGBMRegressor): The trained LightGBM model.
    """

    def __init__(self, params: dict = None, n_splits: int = 5, early_stopping_rounds: int = 150, n_estimators: int = 350):
        """
        Initializes the LightGBMModel with the given parameters.

        Args:
            params (dict, optional): Parameters for LightGBM model. Defaults to None.
            n_splits (int, optional): Number of splits for TimeSeriesSplit. Defaults to 5.
            early_stopping_rounds (int, optional): Rounds of early stopping. Defaults to 150.
            n_estimators (int, optional): Number of boosting iterations. Defaults to 350.
        """
        self.params = params if params is not None else {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'verbose': -1
        }
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.best_iteration = 0
        self.best_score = float('inf')
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fits the LightGBM model using TimeSeriesSplit cross-validation.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.
        """
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        assert isinstance(y, pd.Series), "y must be a pandas Series"

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = lgb.LGBMRegressor(**self.params, n_estimators=self.n_estimators)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(self.early_stopping_rounds)])

            val_pred = model.predict(X_val)
            score = mean_squared_error(y_val, val_pred)

            if score < self.best_score:
                self.best_score = score
                self.best_iteration = model.best_iteration_

        self.model = lgb.LGBMRegressor(**self.params, n_estimators=self.best_iteration)
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts using the trained LightGBM model.

        Args:
            X (pd.DataFrame): Features to predict.

        Returns:
            pd.Series: Predicted values.
        """
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        return pd.Series(self.model.predict(X))


class ProphetsEnsemble:
    """
    An ensemble of Prophet models with different aggregation functions and frequencies.

    Attributes:
    -----------
    freq : str
        Base frequency for the time series data (e.g., 'D' for daily, 'M' for monthly).
    levels : list
        List of levels indicating the different frequencies and aggregation functions to be used.
    h_getter : HolidayBase, optional
        Custom holidays getter to include holidays in the forecasting models.
    prophets_ : dict
        Dictionary to store fitted Prophet models for each level.
    is_fitted_ : bool
        Indicator if the ensemble has been fitted to the data.
    prophet_params : dict
        Additional parameters to pass to the Prophet models.

    """

    def __init__(self, freq: str, levels: list, agg: list, holidays_getter: HolidayBase = None, prophet_params: dict = None):
        self.freq = freq
        self.levels = ['_'.join(x) for x in product(levels, agg)]
        self.h_getter = holidays_getter
        self.prophets_ = dict()
        self.is_fitted_ = False
        self.prophet_params = prophet_params if prophet_params is not None else {}

    @staticmethod
    def _resample(data: pd.DataFrame, freq: str, how: str) -> pd.DataFrame:
        if how not in ['median', 'mean', 'sum', 'count', 'var', 'std', 'min', 'max']:
            raise NotImplementedError(f'Unknown function {how}. Only [median, mean, sum, count, var, std, min, max] are supported.')
        return data.set_index('ds').resample(freq).agg(how).reset_index(drop=False)

    @staticmethod
    def _merge_key_gen(x, level: str) -> str:
        freq = re.sub('[\d]', '', level.split('_')[0])
        if freq == 'H':
            return f'{x.year}-{x.month}-{x.day}-{x.hour}'
        elif freq == 'D':
            return f'{x.year}-{x.month}-{x.day}'
        elif freq == 'M':
            return f'{x.year}-{x.month}'
        elif freq == 'W':
            return f'{x.isocalendar().year}-{x.isocalendar().week}'
        elif freq == 'Q':
            quarter = (x.month - 1) // 3 + 1
            return f'{x.year}-Q{quarter}'
        elif freq == 'Y':
            return f'{x.year}'
        raise NotImplementedError(f'Only [H, D, W, M, Q, Y] are supported. {freq} was received as input!')

    def _get_holidays(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.h_getter is None:
            return None
        holidays = data[['ds']].copy()
        holidays['holiday'] = holidays['ds'].apply(self.h_getter.get)
        return holidays.dropna()

    def _fit_level(self, data: pd.DataFrame, level: str) -> None:
        resampled = self._resample(data, *level.split('_')) if level != self.freq else data.copy()
        
        fb = Prophet(**self.prophet_params, holidays=self._get_holidays(resampled))

        with SuppressOutput():
            fb.fit(resampled)
        self.prophets_[level] = fb

    def _predict_level(self, periods: int, level: str) -> pd.DataFrame:
        fb = self.prophets_[level]
        df = fb.make_future_dataframe(periods=periods, freq=level.split('_')[0])
        forecasts = fb.predict(df)
        forecasts.columns = [f'{x}_{level}' for x in forecasts.columns]
        return forecasts

    def _combine_levels(self, base_df: pd.DataFrame, data: pd.DataFrame, level: str) -> pd.DataFrame:
        key = lambda x: self._merge_key_gen(x, level)
        return (
            base_df.assign(key=base_df['ds'].apply(key))
            .merge(data.assign(key=data[f'ds_{level}'].apply(key)), on='key', how='left')
            .drop(['key', f'ds_{level}'], axis=1)
        )

    @staticmethod
    def _drop_redundant(data: pd.DataFrame) -> pd.DataFrame:
        redundant = [col for col in data.columns if col != 'ds' and 'yhat' not in col and len(data[col].unique()) == 1]
        return data.drop(redundant, axis=1)

    def fit(self, data: pd.DataFrame) -> None:
        for level in [self.freq] + self.levels:
            self._fit_level(data, level)
        self.is_fitted_ = True

    def forecast(self, periods: int) -> pd.DataFrame:
        assert self.is_fitted_, 'Model is not fitted'
        forecasts = [self._predict_level(periods, level) for level in [self.freq] + self.levels]

        forecast = forecasts[0].rename(columns={f'ds_{self.freq}': 'ds', f'yhat_{self.freq}': 'yhat'})
        for level, fore in zip(self.levels, forecasts[1:]):
            forecast = self._combine_levels(forecast, fore, level)

        return self._drop_redundant(forecast)


class BurgerKing:
    """
    A pipeline for forecasting restaurant variables using an ensemble of Prophet models and an Optuna-tuned LightGBM regressor.

    Attributes:
        data (pd.DataFrame): DataFrame containing the restaurant data.
        tqdm (bool): Flag to enable or disable tqdm progress bars.
    """

    def __init__(self, data: pd.DataFrame, freq: str = 'D', periods: int = 180, target: list = None,
                 levels: list = None, agg: list = None, n_splits: int = 5, n_estimators: int = 350,
                 early_stopping_rounds: int = 150, lgb_params: dict = None, prophet_params: dict = None,
                 holidays_getter: HolidayBase = None, tqdm: bool = True):
        self.data = data
        self.freq = freq
        self.periods = periods
        self.target = target if target else ['check_qnty', 'avg']
        self.levels = levels if levels else ['W', 'M']
        self.agg = agg if agg else ['mean', 'median']
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.lgb_params = lgb_params
        self.prophet_params = prophet_params
        self.holidays_getter = holidays_getter if holidays_getter else Russia()
        global tqdm_disabled
        tqdm_disabled = not tqdm

    def process_restaurant_variable(self, rest_id: int, column: str) -> list:
        """
        Processes and forecasts a specific variable for a given restaurant.

        Args:
            rest_id (int): The restaurant ID.
            column (str): The target variable column name.

        Returns:
            list: A list containing the forecast results as DataFrames.
        """
        results = []
        df_rest = self.data[self.data['rest_id'] == rest_id].copy()
        assert not df_rest.empty, f"No data found for restaurant ID {rest_id}"

        df_rest.rename(columns={'day_id': 'ds', column: 'y'}, inplace=True)
        df_rest['ds'] = pd.to_datetime(df_rest['ds'])
        df_rest = df_rest[['ds', 'y']]

        pe = ProphetsEnsemble(freq=self.freq, levels=self.levels, agg=self.agg,
                              holidays_getter=self.holidays_getter, prophet_params=self.prophet_params)
        pe.fit(df_rest)
        pe_forecast = pe.forecast(self.periods)

        assert not pe_forecast.empty, "Prophet ensemble forecast resulted in an empty DataFrame"

        gbt_data = df_rest.merge(pe_forecast, on='ds', how='left')
        assert 'y' in gbt_data.columns, "'y' column missing after merging forecast data"

        X = gbt_data.drop(['ds', 'y'], axis=1)
        y = pd.Series(gbt_data['y'].values)

        lgbm_model = LightGBMModel(params=self.lgb_params, n_splits=self.n_splits,
                                   early_stopping_rounds=self.early_stopping_rounds, n_estimators=self.n_estimators)
        lgbm_model.fit(X, y)

        # Ensure that 'rest_id' and 'y' columns are in the future_gbt DataFrame
        future_dates = pd.date_range(start=df_rest['ds'].max() + pd.Timedelta(days=1), periods=self.periods, freq=self.freq)
        future_data = pd.DataFrame({'ds': future_dates})
        future_forecast = pe_forecast[pe_forecast['ds'].isin(future_dates)]
        future_gbt = future_data.merge(future_forecast, on='ds', how='left')

        assert future_gbt.drop(['ds'], axis=1).shape[0] > 0, "Future GBT data has no rows to predict"

        preds = lgbm_model.predict(future_gbt.drop(['ds'], axis=1))

        df_forecast = future_data.copy()
        df_forecast['pred'] = preds
        df_forecast['rest_id'] = rest_id
        df_forecast['target'] = column

        results.append(df_forecast[['rest_id', 'ds', 'target', 'pred']])
        return results

    def run(self) -> pd.DataFrame:
        """
        Runs the forecasting pipeline for all restaurants and target variables.

        Returns:
            pd.DataFrame: DataFrame containing the forecast results for all restaurants and target variables.
        """
        restaurants = self.data['rest_id'].unique()
        target = self.target

        results = []
        for rest_id in tqdm(restaurants, desc="Restaurants", disable=tqdm_disabled):
            for column in tqdm(target, desc="Variables", disable=tqdm_disabled):
                result = self.process_restaurant_variable(rest_id, column)
                if result:
                    results.extend(result)

        results_df = pd.concat(results, ignore_index=True)
        return results_df
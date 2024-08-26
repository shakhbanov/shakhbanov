import numpy as np
import pandas as pd
from holidays import Russia
from shakhbanov.gapfiller import NaturalGapFiller
from shakhbanov.cleaner import DataCleaner
from sqlalchemy import create_engine
import logging, optuna, os, re, tempfile, shutil, prophet
import lightgbm as lgb
from itertools import product
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm
import multiprocessing

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Отключение логирования для сторонних библиотек
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.setLevel(logging.ERROR)

optuna.logging.set_verbosity(optuna.logging.ERROR)

prophet_logger = logging.getLogger('prophet')
prophet_logger.setLevel(logging.ERROR)

tqdm_disabled = False

n_jobs = 7  # Используем 5 ядер для уменьшения нагрузки и предотвращения переполнения лимита файлов

class SuppressOutput:
    """Контекстный менеджер для подавления вывода в stdout и stderr."""

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

def generate_weather_features(future_dates, weather_forecast_df, historical_weather_data):
    """
    Дополняет будущие даты погодными характеристиками.
    """
    if weather_forecast_df is None or historical_weather_data is None:
        raise ValueError("weather_forecast_df и historical_weather_data не могут быть None.")

    future_weather_df = pd.DataFrame({'ds': future_dates})
    future_weather_df = future_weather_df.merge(weather_forecast_df, on='ds', how='left')

    historical_weather_data['month'] = historical_weather_data['ds'].dt.month
    historical_weather_data['day'] = historical_weather_data['ds'].dt.day
    historical_means = historical_weather_data.groupby(['month', 'day']).mean(numeric_only=True).reset_index()

    def fill_missing(row):
        if pd.isna(row['tempmax']):
            month, day = row['ds'].month, row['ds'].day
            seasonal_data = historical_means[(historical_means['month'] == month) & (historical_means['day'] == day)]
            if not seasonal_data.empty:
                for col in ['tempmax', 'precip']:
                    row[col] = seasonal_data[col].values[0]
        return row

    future_weather_df = future_weather_df.apply(fill_missing, axis=1)
    future_weather_df.fillna(method='ffill', inplace=True)

    return future_weather_df

class LightGBMModel:
    """
    Обертка для обучения и кросс-валидации модели LightGBM с использованием TimeSeriesSplit.
    """

    def __init__(self, params=None, n_splits=5, early_stopping_rounds=150, n_estimators=350):
        self.params = params if params else {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'verbose': -1,
            'verbosity': -1
        }
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.model = None
        self.label_encoders = {}

    def objective(self, trial, X, y, valid_categorical_features):
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': trial.suggest_int('num_leaves', 31, 80),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 60),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'num_threads': n_jobs,
            'verbose': -1,
            'verbosity': -1
        }

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = lgb.LGBMRegressor(**params, n_estimators=self.n_estimators)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(self.early_stopping_rounds)],
                      categorical_feature=valid_categorical_features)

            val_pred = model.predict(X_val)
            scores.append(mean_squared_error(y_val, val_pred))

        return np.mean(scores)

    def fit(self, X, y):
        categorical_features = ['format', 'working_hours']

        valid_categorical_features = []
        for col in categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
                valid_categorical_features.append(col)

        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X, y, valid_categorical_features), n_trials=100, n_jobs=n_jobs)
        self.best_params = study.best_params

        self.model = lgb.LGBMRegressor(**self.best_params, n_estimators=self.n_estimators)
        self.model.fit(X, y, categorical_feature=valid_categorical_features)

    def predict(self, X):
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col])

        return pd.Series(self.model.predict(X))

class ProphetsEnsemble:
    """
    Ансамбль моделей Prophet с различными функциями агрегации и частотами.
    """

    def __init__(self, freq, levels, agg, holidays_getter=None, prophet_params=None,
                 historical_weather_data=None, weather_forecast_df=None):
        self.freq = freq
        self.levels = ['_'.join(x) for x in product(levels, agg)]
        self.holidays_getter = holidays_getter
        self.prophets = {}
        self.is_fitted = False
        self.prophet_params = prophet_params or {}
        self.historical_weather_data = historical_weather_data
        self.weather_forecast_df = weather_forecast_df
        self.cache = {}

    def resample(self, data, freq, how):
        cache_key = (freq, how)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if how not in ['median', 'mean', 'sum', 'count', 'var', 'std', 'min', 'max']:
            raise NotImplementedError(f'Функция {how} не поддерживается.')

        resampled_data = data.set_index('ds').resample(freq).agg(how).reset_index()
        self.cache[cache_key] = resampled_data

        return resampled_data

    @staticmethod
    def merge_key_gen(x, level):
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
        raise NotImplementedError(f'Частота {freq} не поддерживается.')

    def get_holidays(self, data):
        if self.holidays_getter is None:
            return None
        holidays = data[['ds']].copy()
        holidays['holiday'] = holidays['ds'].apply(self.holidays_getter.get)
        return holidays.dropna()

    def objective(self, trial, data, level):
        params = {
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.01, 0.5, log=True),
            'holidays_prior_scale': 20,
            'changepoint_range': 0.95,
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }

        resampled = self.resample(data, *level.split('_')) if level != self.freq else data.copy()
        holidays = self.get_holidays(resampled)
        fb = Prophet(**params, holidays=holidays)

        for regressor in ['tempmax', 'temp', 'tempmin', 'humidity', 'windspeed', 'cloudcover', 'solarenergy', 'usd_lag_2m', 'usd_avg_q']:
            if regressor in resampled.columns:
                fb.add_regressor(regressor)

        with SuppressOutput():
            fb.fit(resampled)

        df = fb.make_future_dataframe(periods=0, freq=level.split('_')[0])
        forecasts = fb.predict(df)
        return mean_squared_error(resampled['y'], forecasts['yhat'])

    def fit_level(self, data, level):
        tmp_dir = tempfile.mkdtemp()  # Создаем временную директорию для хранения файлов Stan
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective(trial, data, level), n_trials=100, n_jobs=n_jobs)
            best_params = study.best_params

            best_params.update({
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False
            })

            resampled = self.resample(data, *level.split('_')) if level != self.freq else data.copy()
            holidays = self.get_holidays(resampled)
            fb = Prophet(**best_params, holidays=holidays)

            for regressor in ['tempmax', 'temp', 'tempmin', 'humidity', 'windspeed', 'cloudcover', 'solarenergy', 'usd_lag_2m', 'usd_avg_q']:
                if regressor in resampled.columns:
                    fb.add_regressor(regressor)

            with SuppressOutput():
                fb.fit(resampled)
            self.prophets[level] = fb

        finally:
            shutil.rmtree(tmp_dir)  # Удаляем временную директорию и все файлы после использования

    def predict_level(self, periods, level):
        fb = self.prophets[level]
        df = fb.make_future_dataframe(periods=periods, freq=level.split('_')[0])

        future_weather_df = generate_weather_features(df['ds'], self.weather_forecast_df, self.historical_weather_data)

        for regressor in future_weather_df.columns:
            if regressor in fb.history.columns:
                df[regressor] = future_weather_df[regressor]

        forecasts = fb.predict(df)
        forecasts.columns = [f'{x}_{level}' for x in forecasts.columns]
        return forecasts

    def combine_levels(self, base_df, data, level):
        key = lambda x: self.merge_key_gen(x, level)
        return (
            base_df.assign(key=base_df['ds'].apply(key))
            .merge(data.assign(key=data.get(f'ds_{level}', pd.Series()).apply(key)), on='key', how='left')
            .drop(['key', f'ds_{level}'], axis=1, errors='ignore')
        )

    @staticmethod
    def drop_redundant(data):
        redundant = [col for col in data.columns if col != 'ds' and 'yhat' not in col and len(data[col].unique()) == 1]
        return data.drop(redundant, axis=1)

    def fit(self, data):
        for level in [self.freq] + self.levels:
            self.fit_level(data, level)
        self.is_fitted = True

    def forecast(self, periods):
        assert self.is_fitted, 'Модель не обучена'
        forecasts = [self.predict_level(periods, level) for level in [self.freq] + self.levels]

        forecast = forecasts[0].rename(columns={f'ds_{self.freq}': 'ds', f'yhat_{self.freq}': 'yhat'})
        for level, fore in zip(self.levels, forecasts[1:]):
            forecast = self.combine_levels(forecast, fore, level)

        return self.drop_redundant(forecast)

class ForecastAdjustment:
    """Модуль для корректировки прогноза с учетом трендов и сезонности."""

    def __init__(self, data, forecast=None, seasonal_period=7):
        self.data = data
        self.forecast = forecast
        self.seasonal_period = seasonal_period

    def calculate_bias(self, errors):
        return np.mean(errors)

    def detect_trend(self, target_data):
        slope, _ = np.polyfit(range(len(target_data)), target_data, 1)
        return slope

    def detect_seasonality(self, target_data):
        decomposition = seasonal_decompose(target_data, period=self.seasonal_period, model='additive')
        return decomposition.seasonal[-self.seasonal_period:]

    def get_last_week_data(self, rest_id, target):
        last_actual_date = self.data[self.data['rest_id'] == rest_id]['ds'].max()
        last_week_dates = pd.date_range(end=last_actual_date, periods=7).strftime('%Y-%m-%d')
        actual_data = self.data[(self.data['rest_id'] == rest_id) & (self.data['ds'].isin(last_week_dates))]

        forecast_data = self.forecast[(self.forecast['rest_id'] == rest_id) & (self.forecast.get('ds', pd.Series()).isin(last_week_dates))] if self.forecast is not None else None

        return actual_data, forecast_data

    def adjust_forecast(self, forecast, rest_id, train, target):
        actual_data, forecast_data = self.get_last_week_data(rest_id, target)

        if actual_data.empty or forecast_data is None or forecast_data.empty:
            return forecast

        errors = actual_data[target].values - forecast_data[target].values
        bias = self.calculate_bias(errors)
        trend = self.detect_trend(train[target].values)
        seasonality = self.detect_seasonality(train[target].values)

        corrected_forecast = forecast + bias + trend * np.arange(len(forecast)) + seasonality

        error_threshold = 2 * np.std(errors)
        significant_errors = np.abs(errors) > error_threshold
        corrected_forecast[significant_errors] += errors[significant_errors] * 0.5

        return corrected_forecast

class BurgerKingPlus:
    """
    Пайплайн для прогнозирования показателей ресторана с использованием ансамбля Prophet и LightGBM.
    """

    def __init__(self, data, forecast=None, freq='D', periods=180, target=None, levels=None, agg=None,
                 n_splits=5, n_estimators=350, early_stopping_rounds=150, lgb_params=None,
                 prophet_params=None, holidays_getter=None, tqdm=True, historical_weather_data=None, weather_forecast_df=None):
        self.data = data
        self.forecast = forecast
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
        self.historical_weather_data = historical_weather_data
        self.weather_forecast_df = weather_forecast_df
        self.adjustment_module = ForecastAdjustment(data, forecast)
        global tqdm_disabled
        tqdm_disabled = not tqdm

    def process_restaurant_variable(self, rest_id, column):
        df_rest = self.data[self.data['rest_id'] == rest_id].copy()
        df_rest.rename(columns={'day_id': 'ds', column: 'y'}, inplace=True)
        df_rest['ds'] = pd.to_datetime(df_rest['ds'])
        df_rest = df_rest[['ds', 'y']]

        pe = ProphetsEnsemble(freq=self.freq, levels=self.levels, agg=self.agg,
                              holidays_getter=self.holidays_getter, prophet_params=self.prophet_params,
                              historical_weather_data=self.historical_weather_data,
                              weather_forecast_df=self.weather_forecast_df)
        pe.fit(df_rest)
        pe_forecast = pe.forecast(self.periods)

        gbt_data = df_rest.merge(pe_forecast, on='ds', how='left')

        X = gbt_data.drop(['ds', 'y'], axis=1)
        y = gbt_data['y']

        lgbm_model = LightGBMModel(params=self.lgb_params, n_splits=self.n_splits,
                                   early_stopping_rounds=self.early_stopping_rounds, n_estimators=self.n_estimators)
        lgbm_model.fit(X, y)

        future_dates = pd.date_range(start=df_rest['ds'].max() + pd.Timedelta(days=1), periods=self.periods, freq=self.freq)
        future_data = pd.DataFrame({'ds': future_dates})
        future_forecast = pe_forecast[pe_forecast['ds'].isin(future_dates)]
        future_gbt = future_data.merge(future_forecast, on='ds', how='left')

        preds = lgbm_model.predict(future_gbt.drop(['ds'], axis=1))

        corrected_preds = self.adjustment_module.adjust_forecast(preds, rest_id, df_rest[['ds', 'y']], target=column)
        df_forecast = future_data.copy()
        df_forecast['pred'] = corrected_preds
        df_forecast['rest_id'] = rest_id
        df_forecast['target'] = column

        return df_forecast[['rest_id', 'ds', 'target', 'pred']]

    def run(self):
        restaurants = self.data['rest_id'].unique()
        target = self.target

        results = []
        for rest_id in tqdm(restaurants, desc="Рестораны", disable=tqdm_disabled):
            for column in tqdm(target, desc="Переменные", disable=tqdm_disabled):
                result = self.process_restaurant_variable(rest_id, column)
                results.append(result)

        results_df = pd.concat(results, ignore_index=True)
        return results_df
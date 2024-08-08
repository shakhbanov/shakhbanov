<p align="center">
  <img src="https://s3.shakhbanov.org/logo.jpeg" alt="Лого" width="200" />
</p>

# shakhbanov_ml

**ML инструменты для временного ряда**

Добро пожаловать в `shakhbanov_ml` - набор инструментов для обработки и анализа временных рядов. Этот пакет включает в себя методы для восстановления пропусков, очистки данных и прогнозирования, специально разработанные для упрощения работы с временными рядами.

---

## 🚀 Установка

```bash
pip install shakhbanov
```

## 🔧 Использование

### 📥 Импортируем библиотеки и загружаем данные

```python
import pandas as pd

from shakhbanov.metrics import wape
from shakhbanov.forecast import BurgerKing
from shakhbanov.cleaner import DataCleaner
from shakhbanov.kalman_gap_filler import KalmanGapFiller

data = pd.read_csv('timeseries.csv')
```

### 🔍 Восстанавливаем пропущенные периоды методом фильтра Калмана

```python
# Инициализация KalmanGapFiller с параметрами
data = KalmanGapFiller(date_column='ds',                   # Колонка с датами
                       group_column='id',                  # Колонка с идентификаторами групп
                       target_columns=['value', 'value2'], # Колонки с целевыми значениями 
                       tqdm=True,                          # Показ прогресса выполнения
                       parallel=True                       # Использование параллельной обработки
                   ).process(data)                         # Обработка данных
```

### 🧹 Очищаем данные от выбросов с учетом тренда

```python
data = DataCleaner(method='mean',      # Метод расчета центральной тенденции ('mean', 'median' или 'mode')
                   weeks=4,            # Количество недель для расчета скользящего среднего
                   window=365,         # Размер окна для скользящего среднего и стандартного отклонения
                   std=2,              # Количество стандартных отклонений для определения выбросов
                   tqdm=True,          # Флаг для использования индикатора прогресса tqdm
                   plot=False,         # Флаг для включения или отключения визуализации
                   num_plot=5,         # Количество объектов для визуализации
                   parallel=True).clean_data(data=data, column=['value', 'value2'], series_id='id')
```

### 📈 Прогнозирование

```python
bk = BurgerKing(
    data=data,                       # Набор данных
    freq='D',                        # Частота временного ряда
    periods=180,                     # Количество периодов для прогнозирования
    target=['check_qnty', 'avg'],    # Целевые переменные для прогнозирования
    levels=['W', 'M'],               # Уровни для ресемплинга ('W' - еженедельно, 'M' - ежемесячно)
    agg=['mean', 'median'],          # Функции агрегации
    n_splits=5,                      # Количество разбиений для TimeSeriesSplit
    n_estimators=350,                # Количество итераций для бустинга в LightGBM
    early_stopping_rounds=150,       # Количество раундов для ранней остановки в LightGBM
    lgb_params={
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'verbose': -1
    },  # Параметры для LightGBM
    prophet_params={
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False
    },  # Параметры для Prophet
    holidays_getter=None,             # Кастомный получатель праздников
    tqdm=True                         # Включить индикаторы прогресса tqdm
)

results_df = bk.run()
```

### 📊 Замеряем качество

```python
wape(data, test)
```

## 📞 Контакты

- 🌐 Сайт: [shakhbanov.org](https://shakhbanov.org)
- 📧 Email: [zurab@shakhbanov.ru](mailto:zurab@shakhbanov.ru)

---

### README.md in English


# shakhbanov_ml

**ML tools for time series**

Welcome to `shakhbanov_ml` - a toolkit for processing and analyzing time series. This package includes methods for gap filling, data cleaning, and forecasting, specifically designed to simplify working with time series data.

---

## 🚀 Installation

```bash
pip install shakhbanov
```

## 🔧 Usage

### 📥 Import Libraries and Load Data

```python
import pandas as pd

from shakhbanov.metrics import wape
from shakhbanov.forecast import BurgerKing
from shakhbanov.cleaner import DataCleaner
from shakhbanov.kalman_gap_filler import KalmanGapFiller

data = pd.read_csv('timeseries.csv')
```

### 🔍 Fill Missing Periods using Kalman Filter

```python
# Initialize KalmanGapFiller with parameters
data = KalmanGapFiller(date_column='ds',
                      group_column='id', 
                      target_columns=['value', 'value2'],
                      tqdm=True, 
                      parallel=True).process(data)
```

### 🧹 Clean Data from Outliers Considering Trend

```python
data = DataCleaner(method='mean',      # Method for calculating central tendency ('mean', 'median' or 'mode')
                   weeks=4,            # Number of weeks for calculating the moving average
                   window=365,         # Window size for moving average and standard deviation
                   std=2,              # Number of standard deviations for identifying outliers
                   tqdm=True,          # Flag to use tqdm progress indicator
                   plot=False,         # Flag to enable or disable visualization
                   num_plot=5,         # Number of items to visualize
                   parallel=True       # Flag to use parallel processing
                ).clean_data(data=data, column=['value', 'value2'], series_id='id')

```

### 📈 Forecasting

```python
bk = BurgerKing(
    data=data,                      # Dataset
    freq='D',                       # Frequency of the time series data
    periods=180,                    # Number of periods to forecast
    target=['check_qnty', 'avg'],   # Target variables to forecast
    levels=['W', 'M'],              # Levels for resampling ('W' for weekly, 'M' for monthly)
    agg=['mean', 'median'],         # Aggregation functions
    n_splits=5,                     # Number of splits for TimeSeriesSplit
    n_estimators=350,               # Number of boosting iterations for LightGBM
    early_stopping_rounds=150,      # Rounds of early stopping for LightGBM
    lgb_params={
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'verbose': -1
    },  # Parameters for LightGBM
    prophet_params={
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False
    },  # Parameters for Prophet
    holidays_getter=None,           # Custom holidays getter
    tqdm=True                       # Enable tqdm progress bars
)

results_df = bk.run()
```

### 📊 Measure Quality

```python
wape(data, test)
```

## 📞 Contacts

- 🌐 Website: [shakhbanov.org](https://shakhbanov.org)
- 📧 Email: [zurab@shakhbanov.ru](mailto:zurab@shakhbanov.ru)
